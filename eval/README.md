# CRM Agent Eval

A CLI test suite that runs a fixed set of 100 CRM questions (mixed
retrieval + task/action, across all the agent's tools) against the running
agent backend, grades each answer with a GPT judge model, and tracks
results over time in a CSV -- one column per run (`test1`, `test2`, `test3`, ...),
so you can see whether a prompt/model/tool change made things better or
worse.

## Files

| File                    | What it is                                                             |
|--------------------------|------------------------------------------------------------------------|
| `questions.csv`          | The 100 fixed questions. `id,category,question,rubric`.               |
| `generate_questions.py`  | Script that produced `questions.csv`. Re-run to regenerate/extend it.  |
| `evaluate.py`            | The eval runner. This is what you actually run.                       |
| `results.csv`            | Generated. One row per question, one column per run.                  |
| `runs/<run>.jsonl`       | Generated. Full transcript + judge reasoning for every question in a run. |

## Question categories

`crm_search` (20), `crm_get` (10), `crm_create` (15), `crm_update` (15),
`crm_delete` (8), `crm_linked_records` (8), `crm_activities` (8),
`crm_contact_action` (6), `crm_research_company` (6), plus a handful of
`multi_step` / `edge_case` / `out_of_scope` questions that mix tools or
probe bad requests (nonexistent records, disallowed doctypes, destructive
bulk deletes, out-of-scope asks).

## How grading works (read this before trusting the scores)

The judge model **does not have access to your live CRM database**, so it
can't check whether a returned lead name or deal value is factually
correct. Instead each question's `rubric` column tells the judge what
*correct behavior* looks like for that request -- did it call the right
kind of tool, did it avoid fabricating specifics it couldn't have known,
did it handle ambiguity/destructive actions sensibly, did it decline
things outside its allowed doctypes. That's what's being scored: tool-use
correctness and answer honesty, not ground-truth accuracy against your
particular CRM data.

Each answer gets:
- `score`: 0-100
- `verdict`: `pass` (>=70), `partial` (40-69), `fail` (<40)

If the agent call itself errors or times out, that's an automatic
`0 (fail)` -- the judge isn't invoked.

## Setup

```bash
cd CRM-Agent-main
pip install -r requirements.txt   # openai + requests are already in there
```

Make sure `.env` (or your environment) has:

```
OPENAI_API_KEY=sk-...          # used for the judge model
MAGMA_API_URL=http://localhost:8005   # wherever server.py is running
EVAL_JUDGE_MODEL=gpt-4o-mini   # optional, this is the default
```

Then start the agent backend as usual (`python server.py` or however you
normally run it) before running the eval.

## Running it

```bash
cd eval

# full run -- creates results.csv, writes column "test1"
python evaluate.py

# next run -- auto-picks "test2" (won't touch the test1 column)
python evaluate.py

# name a run explicitly (re-running with the same name overwrites that column)
python evaluate.py --run-name baseline

# try a handful first before spending judge-model tokens on all 100
python evaluate.py --limit 10
python evaluate.py --category crm_delete
python evaluate.py --ids 1,2,3,42

# re-grade only what a previous run got wrong or partial
python evaluate.py --run-name test3 --retry-failures test2

# see what would run without calling the agent or the judge
python evaluate.py --dry-run

# more parallel requests (default 4), different judge model
python evaluate.py --concurrency 8 --judge-model gpt-4o
```

Each question runs in its own isolated session (`eval-<run>-<id>`) so
answers don't leak context between questions.

## Reading results.csv

```
id,category,question,test1,test2,test3
1,crm_search,"Show me all leads with status 'Open'.",88 (pass),91 (pass),
...
```

Each `testN` cell is `score (verdict)`. Blank means that question wasn't
run in that column (e.g. you used `--category` or `--limit`). For the
full agent reply and the judge's written reasoning behind a given score,
check `runs/<run>.jsonl` -- one JSON object per question, keyed by `id`.

## Extending the question set

Either hand-edit `questions.csv` directly (it's a plain 4-column CSV --
just keep `id` unique), or add entries to the relevant section of
`generate_questions.py` and re-run:

```bash
python generate_questions.py --out questions.csv
```

If you add/remove questions, old `results.csv` columns for question ids
that no longer exist will just be dropped on the next `evaluate.py` run
(it always rebuilds rows from the current `questions.csv`).
