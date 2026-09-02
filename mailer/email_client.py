"""
mailer/email_client.py

Thin SMTP client for the agent's own mailbox. Sending only, for now --
inbound handling (reading replies/new mail and routing a notification to
the right CRM user by role) is a natural extension of this once outbound
is in use, but is out of scope here.

Configure via environment variables (see .env.example):
    SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
    SMTP_FROM_EMAIL, SMTP_FROM_NAME, SMTP_USE_TLS, SMTP_USE_SSL

Works with any standard SMTP provider (Gmail app password, Microsoft 365,
a transactional service like SendGrid/SES SMTP, a self-hosted mail
server, ...) -- there is nothing Frappe/CRM-specific here, this is just
the agent's personal mailbox.
"""
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _split_addresses(value) -> list:
    """Accepts a single address, a comma/semicolon-separated string, or a
    list/tuple of addresses, and always returns a clean list."""
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).replace(";", ",").split(",")
    return [str(a).strip() for a in items if str(a).strip()]


class EmailClient:
    def __init__(self):
        self.host = os.getenv("SMTP_HOST")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME")
        self.password = os.getenv("SMTP_PASSWORD")
        # Falls back to the login username if a separate From isn't given --
        # most providers require these to match (or be an alias of) the
        # authenticated account anyway.
        self.from_email = os.getenv("SMTP_FROM_EMAIL") or self.username
        self.from_name = os.getenv("SMTP_FROM_NAME", "Magma Assistant")
        self.use_ssl = os.getenv("SMTP_USE_SSL", "false").strip().lower() == "true"
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() == "true"
        self.timeout = int(os.getenv("SMTP_TIMEOUT", "20"))

    def is_configured(self) -> bool:
        return bool(self.host and self.username and self.password and self.from_email)

    def send(self, to, subject: str, body: str, cc=None, bcc=None,
              html: bool = False, reply_to: Optional[str] = None) -> dict:
        if not self.is_configured():
            raise RuntimeError(
                "Email is not configured. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD and "
                "SMTP_FROM_EMAIL (see .env.example) before sending."
            )
        to_list = _split_addresses(to)
        cc_list = _split_addresses(cc)
        bcc_list = _split_addresses(bcc)
        if not to_list:
            raise ValueError("At least one 'to' recipient is required.")
        if not subject or not str(subject).strip():
            raise ValueError("A subject is required.")
        if not body or not str(body).strip():
            raise ValueError("A body is required.")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = formataddr((self.from_name, self.from_email))
        msg["To"] = ", ".join(to_list)
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)
        if reply_to:
            msg["Reply-To"] = reply_to
        msg.attach(MIMEText(body, "html" if html else "plain", "utf-8"))

        all_recipients = to_list + cc_list + bcc_list
        context = ssl.create_default_context()

        if self.use_ssl:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=context) as server:
                server.login(self.username, self.password)
                server.sendmail(self.from_email, all_recipients, msg.as_string())
        else:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.sendmail(self.from_email, all_recipients, msg.as_string())

        return {
            "from": self.from_email,
            "to": to_list,
            "cc": cc_list,
            "bcc": bcc_list,
            "subject": subject,
        }


email_client = EmailClient()
