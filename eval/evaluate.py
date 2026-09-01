#!/usr/bin/env python3
"""
evaluate.py -- CLI eval harness for the MagmaAssistance CRM agent.

For each question in questions.csv:
  1. Sends it to the running agent backend (server.py) over the same
     /api/chat HTTP endpoint the CLI and web UI use, in its own isolated
     session so answers don't leak context between questions.
  2. Sends the question, the grading rubric, and the agent's reply to a
     GPT judge model, which returns a score (0-100), a verdict
     (pass/partial/fail), and a short reasoning string.
  3. Writes/updates results.csv: one row per question, one column per
     run (test1, test2, test3, ...). Re-running with a new run creates a
     new column and leaves prior columns untouched, so you can track a
     model/prompt/tool change across runs over time.

Full per-question detail for a run (agent reply + judge reasoning) is
also written to eval/runs/<run_name>.jsonl, since that's too verbose for
the CSV.

Usage:
    # first run -- picks the next free column automatically (test1)
    python evaluate.py

    # explicit run name (also fine to reuse; overwrites that column)
    python evaluate.py --run-name test2

    # try a subset first
    python evaluate.py --limit 10
    python evaluate.py --category crm_delete
    python evaluate.py --ids 1,2,3,42

    # re-run only what a previous run got wrong
    python evaluate.py --run-name test3 --retry-failures test2

    # tune concurrency / models
    python evaluate.py --concurrency 8 --judge-model gpt-4o

Requires:
    - The agent backend (server.py) already running and reachable.
    - OPENAI_API_KEY set (used for the judge; loaded from .env same as
      the rest of this project).

Env vars (all optional, sensible defaults):
    MAGMA_API_URL     backend base URL          (default http://localhost:8005)
    OPENAI_API_KEY    judge model credentials   (required)
    EVAL_JUDGE_MODEL  judge model name          (default gpt-4o-mini)
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from MagnaCLI import MagmaClient, _load_dotenv  # reuse the existing backend client
except ImportError as e:
    print(f"Couldn't import MagmaClient from ../MagnaCLI.py: {e}")
    sys.exit(1)

_load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

try:
    from openai import OpenAI
except ImportError:
    print("This script needs the 'openai' package: pip install openai")
    sys.exit(1)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS = os.path.join(HERE, "questions.csv")
DEFAULT_RESULTS = os.path.join(HERE, "results.csv")
RUNS_DIR = os.path.join(HERE, "runs")

DEFAULT_MAGMA_URL = os.environ.get("MAGMA_API_URL", "http://localhost:8005")
DEFAULT_JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")

JUDGE_SYSTEM_PROMPT = """You are grading a CRM AI agent's response to a user request.

You will be given:
- the user's question
- a rubric describing what a good answer looks like for this category of task
- the agent's actual reply

Important context: you do NOT have access to the live CRM database, so you
cannot verify whether specific record values are factually correct against
real data. Grade the agent on:
  1. Did it take the right kind of action for the request (searched when
     asked to find something, asked for clarification when the request was
     ambiguous, confirmed before/reported clearly on destructive actions,
     declined unsupported operations)?
  2. Did it avoid fabricating specifics (record names, IDs, dates, field
     values) it could not have known, presenting them as fact?
  3. Is the reply well-formed, relevant, and appropriately concise for a
     CRM assistant (not empty, not a refusal for something in scope, not a
     generic non-answer)?
  4. Does it follow the rubric's specific expectations?

Score from 0 to 100:
  90-100 = excellent, fully meets the rubric
  70-89  = good, minor gaps (e.g. slightly vague, missed a small detail)
  40-69  = partial, meets some but not most of the rubric, or fabricates
           a minor detail
  1-39   = poor, wrong approach, fabricates significant details, or
           ignores the request
  0      = no meaningful response / completely off-target / errored out

Respond with ONLY a JSON object, no other text, in this exact shape:
{"score": <integer 0-100>, "verdict": "pass" | "partial" | "fail", "reasoning": "<1-3 sentences>"}

Use "pass" for score >= 70, "partial" for 40-69, "fail" for < 40."""


def load_questions(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_results(path):
    """Returns (fieldnames, {id: row_dict}) or (None, {}) if the file doesn't exist yet."""
    if not os.path.exists(path):
        return None, {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = {row["id"]: row for row in reader}
        return reader.fieldnames, rows


def next_run_name(fieldnames):
    """test1, test2, ... -- picks the first unused testN column."""
    existing = set(fieldnames or [])
    n = 1
    while f"test{n}" in existing:
        n += 1
    return f"test{n}"


def get_agent_reply(base_url, run_name, qid, question, retries=1, timeout=120):
    session_id = f"eval-{run_name}-{qid}"
    client = MagmaClient(base_url, session_id=session_id)
    last_err = None
    for attempt in range(retries + 1):
        try:
            result = client.chat(question)
            reply = result.get("reply", "")
            if not reply:
                return "", "empty reply from agent"
            return reply, None
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2)
    return "", last_err


_judge_lock = threading.Lock()


def judge_reply(openai_client, judge_model, question, rubric, reply, error):
    if error:
        # Agent errored / timed out -- don't bother asking the judge, that's an automatic fail.
        return {"score": 0, "verdict": "fail", "reasoning": f"Agent call failed: {error}"}

    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"AGENT'S REPLY:\n{reply}"
    )
    try:
        completion = openai_client.chat.completions.create(
            model=judge_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        score = max(0, min(100, score))
        verdict = parsed.get("verdict", "fail")
        if verdict not in ("pass", "partial", "fail"):
            verdict = "pass" if score >= 70 else ("partial" if score >= 40 else "fail")
        reasoning = str(parsed.get("reasoning", "")).strip()
        return {"score": score, "verdict": verdict, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0, "verdict": "fail", "reasoning": f"Judge call failed: {e}"}


def run_one(openai_client, judge_model, base_url, run_name, q):
    qid = q["id"]
    reply, error = get_agent_reply(base_url, run_name, qid, q["question"])
    verdict = judge_reply(openai_client, judge_model, q["question"], q["rubric"], reply, error)
    return {
        "id": qid,
        "category": q["category"],
        "question": q["question"],
        "reply": reply,
        "agent_error": error,
        "score": verdict["score"],
        "verdict": verdict["verdict"],
        "reasoning": verdict["reasoning"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="Path to questions.csv")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Path to results.csv (created if missing)")
    parser.add_argument("--url", default=DEFAULT_MAGMA_URL, help=f"Agent backend URL (default {DEFAULT_MAGMA_URL})")
    parser.add_argument("--run-name", default=None, help="Column name for this run, e.g. test3 (default: next free testN)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help=f"OpenAI model used to grade (default {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel questions in flight (default 4)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--category", default=None, help="Only run questions in this category")
    parser.add_argument("--ids", default=None, help="Comma-separated question ids to run, e.g. 1,2,3")
    parser.add_argument("--retry-failures", default=None, metavar="RUN_COLUMN",
                         help="Only run questions whose verdict in this existing run column was 'fail' or 'partial'. "
                              "Questions never graded in that column (blank cell) are skipped, not treated as failures.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without calling the agent or judge")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        print("[error] OPENAI_API_KEY is not set (needed for the judge model). Set it in .env or the environment.")
        sys.exit(1)

    questions = load_questions(args.questions)
    fieldnames, existing_rows = load_results(args.results)

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if args.retry_failures:
        col = args.retry_failures
        if not fieldnames or col not in fieldnames:
            print(f"[error] --retry-failures column '{col}' not found in {args.results}")
            sys.exit(1)
        keep = set()
        for qid, row in existing_rows.items():
            cell = row.get(col, "").strip()
            if not cell:
                continue  # never graded in that column -- not a "failure", just missing data
            if "(fail)" in cell or "(partial)" in cell:
                keep.add(qid)
        questions = [q for q in questions if q["id"] in keep]
        if not questions:
            print(f"No fail/partial rows found in column '{col}' -- nothing to retry.")
            return
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("No questions match the given filters. Nothing to do.")
        return

    run_name = args.run_name or next_run_name(fieldnames)
    print(f"Run: {run_name}  |  {len(questions)} question(s)  |  backend: {args.url}  |  judge: {args.judge_model}")

    if args.dry_run:
        for q in questions:
            print(f"  [{q['id']}] ({q['category']}) {q['question']}")
        print(f"\nWould write results to {args.results}, column '{run_name}'.")
        return

    openai_client = OpenAI()
    os.makedirs(RUNS_DIR, exist_ok=True)
    log_path = os.path.join(RUNS_DIR, f"{run_name}.jsonl")

    results_by_id = {}
    completed = 0
    total = len(questions)
    log_lock = threading.Lock()

    with open(log_path, "w", encoding="utf-8") as log_fh, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:

        futures = {
            pool.submit(run_one, openai_client, args.judge_model, args.url, run_name, q): q
            for q in questions
        }
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {
                    "id": q["id"], "category": q["category"], "question": q["question"],
                    "reply": "", "agent_error": str(e), "score": 0, "verdict": "fail",
                    "reasoning": f"Unhandled error: {e}",
                }
            results_by_id[res["id"]] = res
            completed += 1
            print(f"  [{completed}/{total}] id={res['id']:<4} {res['verdict']:<7} score={res['score']:<3} ({res['category']})")

            with log_lock:
                log_fh.write(json.dumps({
                    "run": run_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **res,
                }, ensure_ascii=False) + "\n")
                log_fh.flush()

    # ---- merge into results.csv --------------------------------------
    all_ids = [q["id"] for q in load_questions(args.questions)]
    base_questions = {q["id"]: q for q in load_questions(args.questions)}

    merged_fieldnames = ["id", "category", "question"]
    prior_test_cols = [f for f in (fieldnames or []) if f.startswith("test")]
    for col in prior_test_cols:
        if col not in merged_fieldnames:
            merged_fieldnames.append(col)
    if run_name not in merged_fieldnames:
        merged_fieldnames.append(run_name)

    with open(args.results, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=merged_fieldnames)
        writer.writeheader()
        for qid in all_ids:
            base = base_questions[qid]
            old = existing_rows.get(qid, {})
            row = {"id": qid, "category": base["category"], "question": base["question"]}
            for col in prior_test_cols:
                row[col] = old.get(col, "")
            if qid in results_by_id:
                r = results_by_id[qid]
                row[run_name] = f"{r['score']} ({r['verdict']})"
            else:
                row[run_name] = old.get(run_name, "")
            writer.writerow(row)

    # ---- summary --------------------------------------------------------
    scores = [r["score"] for r in results_by_id.values()]
    verdicts = [r["verdict"] for r in results_by_id.values()]
    avg = sum(scores) / len(scores) if scores else 0
    pass_n = verdicts.count("pass")
    partial_n = verdicts.count("partial")
    fail_n = verdicts.count("fail")

    print(f"\n{run_name} summary: avg={avg:.1f}  pass={pass_n}  partial={partial_n}  fail={fail_n}  (of {len(results_by_id)})")
    print(f"Results written to {args.results}")
    print(f"Full transcripts + judge reasoning: {log_path}")


if __name__ == "__main__":
    main()
