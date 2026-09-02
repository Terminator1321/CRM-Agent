"""
mailer/email_tools.py

Agent-facing tool for sending email from the agent's own mailbox
(configured via SMTP_* env vars -- see .env.example). Sending only for
now; reading a shared inbox and notifying the right CRM user by role
when a client or teammate replies is a natural next step once this is
in use, not built yet.

Recipient addresses are never guessed. Resolve them from live CRM data
first:
  - team member  -> crm_task_assignment_options(search_text=name) -> "email"
  - CRM contact  -> crm_search(doctype="Contact", search_text=name) -> "email_id"
                    or crm_contact_action(action="search_emails", value=name)
then pass the confirmed address(es) into send_email.
"""
import json
from typing import Optional

from langchain_core.tools import tool

from .email_client import email_client


def out(value):
    return json.dumps(value, ensure_ascii=False, default=str)


@tool
def send_email(to: str, subject: str, body: str, cc: Optional[str] = None,
                bcc: Optional[str] = None, html: bool = False) -> str:
    """Send an email from the agent's own mailbox to one or more recipients.

    Works for clients and team members alike -- whatever address(es) you
    pass in `to` (comma-separated for more than one). Before calling this:
      1. Resolve the real email address from live CRM data -- never invent
         or guess one. For a team member, call
         crm_task_assignment_options(search_text=<name>) and read "email"
         from the matching option. For a client/contact, call
         crm_search(doctype="Contact", search_text=<name>) (field
         "email_id") or crm_contact_action(action="search_emails", value=<name>).
      2. Show the user the resolved recipient(s), the subject, and a short
         summary of the body, and get their explicit confirmation before
         sending. Sending mail is an outbound, irreversible action -- treat
         it with the same confirm-before-you-do-it care as a CRM write.

    `cc` and `bcc` take a single address or a comma-separated list, same as
    `to`. Set `html=True` only when `body` is actual HTML markup; otherwise
    it is sent as plain text.
    """
    try:
        result = email_client.send(to=to, subject=subject, body=body, cc=cc, bcc=bcc, html=html)
        return out({"status": "sent", **result})
    except (ValueError, RuntimeError) as exc:
        return out({"status": "error", "error": str(exc)})


EMAIL_TOOLS = [send_email]
