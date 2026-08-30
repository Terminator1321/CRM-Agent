"""
LLM.py

A reusable LLM wrapper class around OpenAI's chat completions API.
Can be imported and used in other projects, or run directly as a CLI chat.

Requirements:
    pip install requests python-dotenv pymupdf

.env file should contain:
    OPENAI_API_KEY=your_key_here
"""

import os
import requests
import base64
import json
import logging
from dotenv import load_dotenv
import fitz  # PyMuPDF: For converting PDF pages to PNG images

try:
    from langsmith import traceable
except ImportError:
    # LangSmith is optional -- fall back to a no-op decorator so this
    # module still works with tracing simply turned off, instead of
    # requiring the package.
    def traceable(*t_args, **t_kwargs):
        def decorator(fn):
            return fn
        if t_args and callable(t_args[0]) and not t_kwargs:
            return t_args[0]
        return decorator

load_dotenv()

logger = logging.getLogger("llm-ocr")

# Specialized Prompts for Multi-Agent LangGraph Workflow

INTENT_SYSTEM_PROMPT = (
    "You are a strict intent-routing agent for a standalone Frappe CRM assistant.\n"
    "Classify messages as chitchat, crm_query, crm_write, or web_search and extract company/person/record entities.\n"
    "crm_query means reading/searching existing CRM records. crm_write means creating, updating, or deleting CRM records. web_search means explicit internet research.\n"
    "Output only structured JSON. Never answer the user directly."
)

RESEARCH_SYSTEM_PROMPT = (
    "You are a dedicated web research agent for a CRM. Gather evidence-backed company profiles.\n"
    "Use only web_search, web_fetch_page, web_company_search, and web_company_extract.\n"
    "Find the official website, description, industry, headquarters, contact email/phone, social profiles, and people when requested.\n"
    "Never guess contact details. Try the official contact/about pages before declaring a field NOT FOUND. Include source URLs for verified facts."
)

PROPOSAL_SYSTEM_PROMPT = (
    "You are a CRM proposal and validation agent. Review web research and CRM context, prevent duplicates, "
    "and present a concise proposed CRM change. Never create the record yourself; ask for confirmation before a write. "
    "For researched data, include source URLs and clearly mark anything not found or uncertain."
)

GENERAL_CRM_PROMPT = (
    "You are Magma, a professional AI assistant for a standalone Frappe CRM. "
    "Use CRM tools for existing CRM data and web tools for internet research. "
    "Use crm_metadata when you need live field definitions. Never invent CRM fields or company contact details. "
    "Before destructive or consequential CRM writes, obtain explicit user confirmation. "
    "\n\n"
    "Company research is automatic, not optional: whenever the user asks you to create or update a "
    "Lead, Deal, CRM Organization, or Contact that names a company, and that company isn't already a "
    "CRM Organization record with a website/email/phone on file, immediately call "
    "crm_research_company yourself (do this without being asked -- don't wait for the user to say "
    "'search for it'). This applies even when the user's message already contains everything Frappe "
    "requires to create the record (e.g. just a first name, last name, and company is already enough "
    "for a valid Lead) -- having the minimum required fields is NOT the same as having researched the "
    "company, and is never a reason to skip this step. Treat 'create/add a lead|deal|contact for "
    "<person> from/at <company>' as a two-step job every time: research the company first, THEN "
    "create the record -- never go straight to crm_create on the first turn.\n"
    "crm_research_company chains web_company_search -> web_company_extract -> web_search internally, "
    "so one call gets you the official website, description, industry, and -- when they're actually "
    "published on the company's own pages -- email, phone, WhatsApp, address, and social profiles. "
    "Skip this step only if the user already gave you all the relevant details themselves in their "
    "message, or explicitly says not to look it up.\n"
    "Before writing anything to the CRM, show the user what you found as a short summary (website, "
    "email, phone, industry, one-line description -- omit fields that came back NOT FOUND, never "
    "invent a value for them) and ask them to confirm it's correct, e.g. 'Found this for Magna Data "
    "Company: site X, email Y, phone Z -- look right? I'll use this for the record.' Only proceed "
    "with crm_create/crm_update once they confirm or correct it. If research turns up conflicting "
    "candidates (e.g. web_company_search returns several similarly-named companies), show the "
    "shortlist and ask which one before extracting further, per web_company_search's own guidance.\n\n"
    "When to search the web more generally: if the user mentions a person you don't have CRM data on, "
    "or asks you to search, look up, or find something explicitly, use the appropriate web tool "
    "rather than answering from memory.\n\n"
    "After every successful CRM create, update, or delete, end the reply with a frontend-friendly "
    "action block in exactly this format (one option per line, no Markdown bullets):\n"
    "Next actions:\n[A] <specific action>\n[B] <specific action>\n[C] <specific action>\n"
    "Use exactly three short, record-specific options. For example, after creating a Lead: "
    "[A] Update this lead; [B] Create a deal for this lead; [C] Delete this lead. After creating a "
    "Deal: update the deal, add a note or call log, or create a follow-up task. After creating a "
    "Task: update it, mark it done, or reassign it. After creating a Note or Call Log: edit it, "
    "create a follow-up task, or view related activity. After a delete, offer relevant non-destructive "
    "next actions such as search records, create a replacement, or view related activity. Do not add "
    "this block after a failed action, while awaiting confirmation, or for ordinary chitchat. The "
    "frontend can render the literal A/B/C lines as action buttons.\n\n"
    "If a crm_create/crm_update result has \"error\": \"invalid_link_fields\", handle each problem by "
    "its \"creatable\" flag, don't just relay the raw error:\n"
    "  - creatable=false (a fixed status/workflow list, e.g. CRM Lead Status, CRM Deal Status): "
    "NEVER offer to create a new option. Pick the closest sensible match from valid_options yourself "
    "(e.g. a fresh record usually maps to the first/default-sounding option like 'New' or "
    "'Qualification') and retry. Only ask the user if genuinely no option is a reasonable match.\n"
    "  - creatable=true (open reference data, e.g. CRM Organization, CRM Industry, CRM Territory, "
    "CRM Lead Source, CRM Lost Reason): this is a missing dependency, not a mistake. Ask the user a "
    "short, direct confirmation ('BJP doesn't exist as an Organization yet -- create it and continue?'). "
    "If they confirm, call crm_create for that linked doctype yourself, then retry the original call "
    "with the same value -- don't make them create it manually. If they decline or you're unsure what "
    "minimal data the new record needs, ask.\n"
    "This same ask-before-fixing-a-dependency / just-report-a-real-error split applies generally: silently "
    "retry or fix things you can safely resolve from data already available (bad but guessable status, a "
    "missing reference record the user just confirmed), but surface genuine errors (permission denied, "
    "malformed input, conflicting required fields) plainly instead of guessing around them.\n\n"
    "Before creating or updating a Deal, Task, Note, or Call Log tied to a person or lead the user "
    "referred to by name (especially if misspelled, partial, or from earlier in the conversation), "
    "ALWAYS run a fresh crm_search with search_text on CRM Lead (and Contact/CRM Organization if "
    "relevant) first -- never reuse a name or record ID from earlier turns without re-verifying it "
    "still matches what the user means now. If search returns exactly one clear match, proceed. If "
    "it returns multiple candidates, rank them before asking: prefer active/open statuses (New, "
    "Contacted, Nurture, Qualified, Converted) over closed-negative ones (Lost, Unqualified, Junk), "
    "since the user is unlikely to want a deal against a lead that's already lost or disqualified. "
    "Present the ranked candidates (name, organization, status) and ask the user to pick one before "
    "creating anything.\n\n"
    "Task assignment is mandatory for every new CRM Task. Before calling crm_create for a task, always "
    "call crm_task_assignment_options. If the user named an assignee, search for that person first and "
    "show the exact matching enabled CRM user for confirmation. If they did not name one, show the "
    "recommended enabled user (fewest active tasks; Done and Canceled do not count) and ask whether to "
    "assign the task to them. Do not silently assign anyone. If they decline, show the available users "
    "and ask who should receive it; keep asking until they select and confirm an enabled user.\n"
    "If the requested assignee does not exist, say so plainly and ask: 'That person is not an enabled "
    "CRM user. Do you want me to create an account for them?' Do not create an account merely because "
    "they gave a new name. After explicit confirmation, collect and repeat back the person's first name, "
    "email, optional last name, and whether to send a welcome email; only then call crm_create_user. "
    "After it succeeds, confirm the account and task assignment, then create the task with assigned_to "
    "set to the returned user name. Never set roles, passwords, or privileged account settings."
)

class LLM:

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", system_prompt: str = GENERAL_CRM_PROMPT, temperature: float = 0.1, base_url: str = "https://api.openai.com/v1/chat/completions"):
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        env_openai_key = os.environ.get("OPENAI_API_KEY")
        key = api_key or openrouter_key or env_openai_key

        is_openrouter = bool(openrouter_key) or (key and key.startswith("sk-or-v1-"))

        if is_openrouter:
            self.api_key = openrouter_key or key
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"
            self.model_name = model if "/" in model else f"openai/{model}"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8050",
                "X-Title": "MagmaAssistance",
            }
        else:
            self.api_key = key
            if not self.api_key:
                raise ValueError("No API key provided. Set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file.")
            self.model_name = model
            self.base_url = base_url
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.history = [{"role": "system", "content": self.system_prompt}]

    def set_system_prompt(self, system_prompt: str, reset_history: bool = True):
        self.system_prompt = system_prompt
        if reset_history:
            self.history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.history[0] = {"role": "system", "content": self.system_prompt}

    @traceable(name="LLM.chat", run_type="llm")
    def chat(self, user_input: str, remember: bool = True):
        messages = self.history + [{"role": "user", "content": user_input}]

        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = requests.post(
            self.base_url,
            json=data,
            headers=self.headers
        )

        reply = response.json()['choices'][0]['message']['content']

        if remember:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": reply})

        return reply

    def chat_stream(self, user_input: str, remember: bool = True):
        messages = self.history + [{"role": "user", "content": user_input}]
        data = {"model": self.model_name, "messages": messages, "temperature": self.temperature, "stream": True}
        full = ""
        with requests.post(self.base_url, json=data, headers=self.headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if not line.startswith("data: "): continue
                payload = line[6:].strip()
                if payload == "[DONE]": break
                try: chunk = json.loads(payload)
                except Exception: continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    full += delta
                    yield delta
        if remember:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": full})

    def reset(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    # =====================================================================
    # MULTI-FORMAT OCR VISION METHOD (PDF + IMAGE SUPPORT)
    # =====================================================================
    # =====================================================================
    # GENERAL-PURPOSE DOCUMENT READER (ANY PDF / IMAGE, NOT JUST POs)
    # =====================================================================
    @traceable(name="LLM._vision_transcribe_image", run_type="llm")
    def _vision_transcribe_image(self, data_url: str) -> str:
        """Helper: sends one image to GPT-4o Vision and returns a plain
        transcription of everything visible on it (tables rendered as
        markdown tables). Used by extract_document_text() for scanned
        pages / plain images -- separate from the strict PO JSON prompt
        used in extract_po_data_from_document()."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Transcribe ALL text visible in this document image "
                        "exactly as it appears, preserving reading order. "
                        "Render any tables as markdown tables. Do not "
                        "summarize, comment, or add anything not present in "
                        "the image."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this document."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.0,
        }
        response = requests.post(self.base_url, json=payload, headers=self.headers)
        response.raise_for_status()
        res_json = response.json()
        message = res_json["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            refusal = message.get("refusal")
            logger.warning("Vision transcription returned no content: %s", refusal)
            return f"[Could not read this page: {refusal or 'no content returned'}]"
        return content.strip()

    @traceable(name="LLM.extract_document_text", run_type="chain")
    def extract_document_text(self, file_bytes: bytes, mime_type: str, max_pages: int = 20) -> dict:
        """
        Reads ANY PDF or image and returns its full text content, so the
        user can later ask free-form questions about it in chat. This is
        intentionally separate from extract_po_data_from_document(),
        which keeps doing the strict Purchase-Order JSON extraction for
        the CRM flow -- that method is untouched.

        Strategy for PDFs: try PyMuPDF's native text layer first (fast,
        free, no API call needed) since most PDFs are digital-native.
        Only fall back to GPT-4o Vision OCR, page by page, for pages
        whose native text comes back empty/near-empty (i.e. scanned or
        image-only pages).

        Returns: {"text": str, "page_count": int, "pages_read": int,
                  "method": "native" | "vision" | "mixed"}
        """
        try:
            if "pdf" in mime_type.lower():
                try:
                    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                except Exception as e:
                    raise RuntimeError(f"Could not open PDF (it may be corrupted): {e}")

                if pdf_doc.is_encrypted:
                    if not pdf_doc.authenticate(""):
                        pdf_doc.close()
                        raise RuntimeError(
                            "This PDF is password-protected and cannot be read."
                        )

                if len(pdf_doc) == 0:
                    pdf_doc.close()
                    raise RuntimeError("This PDF has no pages.")

                page_count = len(pdf_doc)
                pages_to_read = min(page_count, max_pages)
                if page_count > max_pages:
                    logger.info(
                        "PDF has %d pages; only reading the first %d.",
                        page_count, max_pages
                    )

                page_texts = []
                used_native = False
                used_vision = False

                for page_index in range(pages_to_read):
                    page = pdf_doc[page_index]
                    native_text = page.get_text().strip()

                    if len(native_text) >= 40:
                        # Real digital text layer -- use it directly, no API call.
                        page_texts.append(native_text)
                        used_native = True
                    else:
                        # Likely a scanned/image-only page -- OCR just this page.
                        used_vision = True
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        b64_str = base64.b64encode(img_bytes).decode("utf-8")
                        data_url = f"data:image/png;base64,{b64_str}"
                        page_texts.append(self._vision_transcribe_image(data_url))

                pdf_doc.close()

                method = "mixed" if (used_native and used_vision) else ("vision" if used_vision else "native")
                full_text = "\n\n".join(
                    f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(page_texts)
                )

                return {
                    "text": full_text,
                    "page_count": page_count,
                    "pages_read": pages_to_read,
                    "method": method,
                }

            else:
                # Plain image -- Vision OCR is the only option.
                b64_file = base64.b64encode(file_bytes).decode("utf-8")
                data_url = f"data:{mime_type};base64,{b64_file}"
                text = self._vision_transcribe_image(data_url)
                return {"text": text, "page_count": 1, "pages_read": 1, "method": "vision"}

        except Exception as e:
            logger.exception("Error extracting general document text")
            raise RuntimeError(f"Document text extraction failed: {str(e)}")

    @traceable(name="LLM.ask_about_document", run_type="llm")
    def ask_about_document(self, document_text: str, question: str, max_chars: int = 40000) -> str:
        """One-shot Q&A over already-extracted document text. Does NOT
        touch self.history, so it never pollutes normal chat memory --
        use this (or inject the text into a chat turn) for 'what does
        this document say about X' style questions."""
        trimmed = document_text[:max_chars]
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the document "
                    "content below. If the answer isn't in the document, "
                    "say so plainly -- never invent information.\n\n"
                    "--- DOCUMENT CONTENT ---\n" + trimmed
                ),
            },
            {"role": "user", "content": question},
        ]
        data = {"model": self.model_name, "messages": messages, "temperature": 0.0}
        response = requests.post(self.base_url, json=data, headers=self.headers)
        response.raise_for_status()
        res_json = response.json()
        message = res_json["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            refusal = message.get("refusal")
            return f"Could not answer: {refusal or 'no content returned from model'}"
        return content.strip()


def run_cli():
    print("LLM CLI - OpenAI. Type 'exit' or 'quit' to stop, 'reset' to clear history.\n")
    llm = LLM()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if user_input.lower() == "reset":
            llm.reset()
            print("(history cleared)\n")
            continue

        try:
            reply = llm.chat(user_input)
            print(f"AI: {reply}\n")
        except Exception as e:
            print(f"[Error contacting OpenAI: {e}]\n")


if __name__ == "__main__":
    run_cli()
