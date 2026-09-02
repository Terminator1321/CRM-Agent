import imaplib
import email
import email.utils
import os
import re
import sqlite3
import hashlib
import time
import smtplib
import uuid
from collections import defaultdict
from typing import Optional
from email.header import decode_header
from email.utils import parsedate_to_datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "emails.db"


class EmailReader:
    """
    Connects to Gmail via IMAP, fetches emails (all or unread),
    builds conversation threads, persists to SQLite, and exposes
    agent-ready context strings.
    """

    def __init__(self, db_name: str = DB_NAME):
        self.db_name = db_name
        self.mail: imaplib.IMAP4_SSL | None = None
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            message_id  TEXT PRIMARY KEY,
            thread_id   TEXT,
            parent_id   TEXT,
            chain_index INTEGER,
            sender      TEXT,
            receiver    TEXT,
            subject     TEXT,
            body        TEXT,
            date        TEXT,
            is_unread   INTEGER DEFAULT 1
        )
        """)
        # Migration: add is_unread if DB was created before this column existed
        existing = {row[1] for row in cur.execute("PRAGMA table_info(emails)")}
        if "is_unread" not in existing:
            cur.execute("ALTER TABLE emails ADD COLUMN is_unread INTEGER DEFAULT 1")

        # Tracks a question sent to a human (e.g. "create a lead for X?")
        # that's waiting on their email reply before any action is taken.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pending_actions (
            question_message_id TEXT PRIMARY KEY,
            action_type          TEXT,
            payload_json         TEXT,
            asked_to             TEXT,
            question_subject     TEXT,
            ref_token            TEXT,
            status               TEXT DEFAULT 'pending',
            created_at           TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved_at          TEXT
        )
        """)
        existing_pending = {row[1] for row in cur.execute("PRAGMA table_info(pending_actions)")}
        if "question_subject" not in existing_pending:
            cur.execute("ALTER TABLE pending_actions ADD COLUMN question_subject TEXT")
        if "ref_token" not in existing_pending:
            cur.execute("ALTER TABLE pending_actions ADD COLUMN ref_token TEXT")
        conn.commit()
        conn.close()

    def connect(self) -> "EmailReader":
        user = os.getenv("GMAIL_USERNAME")
        pwd  = os.getenv("GMAIL_PASSWORD")

        if not user or not pwd:
            raise ValueError(
                "Set GMAIL_USERNAME and GMAIL_APP_PASSWORD in your .env\n"
                "App Password: https://myaccount.google.com/apppasswords"
            )

        self.mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        self.mail.login(user, pwd)
        print(f"[EmailReader] Connected as {user}")
        return self

    def disconnect(self):
        if self.mail:
            try:
                self.mail.logout()
            except Exception:
                pass
            self.mail = None
            print("[EmailReader] Disconnected")


    @staticmethod
    def _decode_mime(value: str) -> str:
        if not value:
            return ""
        parts = decode_header(value)
        out = ""
        for text, charset in parts:
            if isinstance(text, bytes):
                out += text.decode(charset or "utf-8", errors="replace")
            else:
                out += text
        return out.strip()

    @staticmethod
    def _normalize_subject(subject: str) -> str:
        if not subject:
            return ""
        subject = subject.lower()
        subject = re.sub(r"^(re:|fwd:|fw:)\s*", "", subject)
        return subject.strip()

    @staticmethod
    def _extract_body(msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    charset = part.get_content_charset() or "utf-8"
                    return part.get_payload(decode=True).decode(charset, errors="replace")
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
        return ""


    def fetch_all(self, limit: int = 200) -> list[dict]:
        """Fetch last `limit` emails from INBOX regardless of read status."""
        self.mail.select("INBOX")#type: ignore
        _, data = self.mail.search(None, "ALL")#type: ignore
        ids = data[0].split()[-limit:]
        return self._parse_ids(ids, is_unread=False)

    def fetch_unread(self) -> list[dict]:
        """
        Fetch only UNSEEN emails. Marks them as Seen on server after fetch.
        Returns parsed message dicts with is_unread=1.
        """
        self.mail.select("INBOX")#type: ignore
        _, data = self.mail.search(None, "UNSEEN")#type: ignore

        if not data[0]:
            return []

        ids = data[0].split()
        print(f"[EmailReader] {len(ids)} unread email(s) found")
        messages = self._parse_ids(ids, is_unread=True)

        for msg_id in ids:
            self.mail.store(msg_id, "+FLAGS", "\\Seen")#type: ignore

        return messages

    def _parse_ids(self, ids: list, is_unread: bool = False) -> list[dict]:
        messages = []
        for msg_id in ids:
            _, msg_data = self.mail.fetch(msg_id, "(RFC822)")#type: ignore
            raw = msg_data[0][1]#type: ignore
            msg = email.message_from_bytes(raw)#type: ignore

            date_raw = msg.get("Date", "")
            try:
                parsed_date = parsedate_to_datetime(date_raw) if date_raw else None
            except Exception:
                parsed_date = None

            references = self._decode_mime(msg.get("References", "")).split()

            messages.append({
                "message_id":  self._decode_mime(msg.get("Message-ID", "")),
                "subject":     self._decode_mime(msg.get("Subject", "")),
                "from":        self._decode_mime(msg.get("From", "")),
                "to":          self._decode_mime(msg.get("To", "")),
                "date":        parsed_date,
                "raw_date":    date_raw,
                "in_reply_to": self._decode_mime(msg.get("In-Reply-To", "")),
                "references":  references,
                "body":        self._extract_body(msg),
                "is_unread":   int(is_unread),
            })

        return messages


    def build_threads(self, messages: list[dict]) -> list[dict]:
        """
        Groups messages into threads by:
        1. In-Reply-To / References headers
        2. Normalized subject fallback
        Assigns thread_id, parent_id, chain_index.
        """
        by_id       = {m["message_id"]: m for m in messages if m["message_id"]}
        subject_map = {}

        def get_thread_id(msg):
            parent = msg["in_reply_to"] or (
                msg["references"][-1] if msg["references"] else None
            )
            if parent and parent in by_id:
                return by_id[parent].get("thread_id")

            norm = self._normalize_subject(msg["subject"])
            if norm in subject_map:
                return subject_map[norm]

            tid = hashlib.md5(norm.encode()).hexdigest()
            subject_map[norm] = tid
            return tid

        for msg in messages:
            msg["thread_id"] = get_thread_id(msg)
            parent = msg["in_reply_to"] or (
                msg["references"][-1] if msg["references"] else ""
            )
            msg["parent_id"] = parent if parent in by_id else ""

        threads = defaultdict(list)
        for m in messages:
            threads[m["thread_id"]].append(m)

        final = []
        for msgs in threads.values():
            msgs.sort(key=lambda x: x["date"] or 0)
            for i, m in enumerate(msgs):
                m["chain_index"] = i + 1
            final.extend(msgs)

        return final


    def save(self, messages: list[dict]):
        conn = sqlite3.connect(self.db_name)
        cur  = conn.cursor()
        for m in messages:
            cur.execute("""
            INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                m["message_id"],
                m["thread_id"],
                m["parent_id"],
                m["chain_index"],
                m["from"],
                m["to"],
                m["subject"],
                m["body"],
                str(m["date"]),
                m.get("is_unread", 0),
            ))
        conn.commit()
        conn.close()

    def get_thread_by_id(self, message_id: str) -> list[tuple]:
        conn = sqlite3.connect(self.db_name)
        cur  = conn.cursor()
        cur.execute("""
        SELECT * FROM emails
        WHERE thread_id = (SELECT thread_id FROM emails WHERE message_id = ?)
        ORDER BY chain_index
        """, (message_id,))
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_unread_from_db(self, limit: int = 20) -> list[tuple]:
        """Pull saved unread rows from DB (for agent context without re-fetching)."""
        conn = sqlite3.connect(self.db_name)
        cur  = conn.cursor()
        cur.execute("""
        SELECT sender, receiver, subject, body, date, chain_index, thread_id, message_id
        FROM emails WHERE is_unread = 1
        ORDER BY date DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows


    def build_agent_context(self, message_id=None, limit: int = 10) -> str:
        """
        Returns a formatted context string ready to inject into an LLM prompt.

        message_id given  → full thread for that email
        message_id=None   → latest `limit` unread emails with their threads

        Format per email:
            [Thread N | Email M/K]
            Subject : ...
            From    : ...
            To      : ...
            Date    : ...
            ---
            <body (truncated to 800 chars)>
            ============================================================
        """
        conn = sqlite3.connect(self.db_name)
        cur  = conn.cursor()

        if message_id:
            cur.execute("""
            SELECT sender, receiver, subject, body, date, chain_index, thread_id
            FROM emails
            WHERE thread_id = (SELECT thread_id FROM emails WHERE message_id = ?)
            ORDER BY chain_index
            """, (message_id,))
        else:
            cur.execute("""
            SELECT sender, receiver, subject, body, date, chain_index, thread_id
            FROM emails WHERE is_unread = 1
            ORDER BY date DESC LIMIT ?
            """, (limit,))

        rows = cur.fetchall()
        conn.close()

        if not rows:
            return "No emails found."

        thread_groups = defaultdict(list)
        for row in rows:
            thread_groups[row[6]].append(row)

        parts = []
        for t_idx, (_, emails) in enumerate(thread_groups.items(), 1):
            k = len(emails)
            for row in emails:
                sender, receiver, subject, body, date, chain_index, _ = row
                body_preview = body[:800].strip() + ("..." if len(body) > 800 else "")
                parts.append(
                    f"[Thread {t_idx} | Email {chain_index}/{k}]\n"
                    f"Subject : {subject}\n"
                    f"From    : {sender}\n"
                    f"To      : {receiver}\n"
                    f"Date    : {date}\n"
                    f"---\n"
                    f"{body_preview}\n"
                    f"{'=' * 60}"
                )

        return "\n\n".join(parts)

    def sync_all(self, limit: int = 200):
        """Full inbox sync: fetch → thread → save."""
        msgs = self.fetch_all(limit)
        msgs = self.build_threads(msgs)
        self.save(msgs)
        print(f"[EmailReader] Synced {len(msgs)} emails")

    def reply_to_email(self, message_id: str, reply_body: str, attachment_path= None) -> tuple[bool, str]:
        """
        Reply to an email with optional attachment.
        
        Args:
            message_id: The Message-ID of the email to reply to
            reply_body: The text body of the reply
            attachment_path: Optional path to a file to attach
        
        Returns:
            (success: bool, message: str)
        """
        user = os.getenv("GMAIL_USERNAME")
        app_pwd = os.getenv("GMAIL_PASSWORD")
        
        if not user or not app_pwd:
            return False, "GMAIL_USERNAME or GMAIL_PASSWORD not set in .env"
        
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute("""
        SELECT sender, receiver, subject, message_id, thread_id
        FROM emails WHERE message_id = ?
        """, (message_id,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return False, f"Email with message_id '{message_id}' not found in database"
        
        original_sender, original_receiver, subject, orig_msg_id, thread_id = row
        
        try:
            msg = MIMEMultipart()
            msg["From"] = user
            msg["To"] = original_sender  
            msg["Subject"] = f"Re: {subject}" if not subject.lower().startswith("re:") else subject
            msg["In-Reply-To"] = orig_msg_id
            msg["References"] = orig_msg_id
            
            msg.attach(MIMEText(reply_body, "plain"))
            
            if attachment_path:
                if not os.path.isfile(attachment_path):
                    return False, f"Attachment file not found: {attachment_path}"
                
                try:
                    with open(attachment_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    filename = os.path.basename(attachment_path)
                    part.add_header("Content-Disposition", f"attachment; filename= {filename}")
                    msg.attach(part)
                    print(f"[EmailReader] Attached file: {filename}")
                except Exception as exc:
                    return False, f"Failed to attach file: {exc}"

            try:
                smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                smtp.login(user, app_pwd)
                smtp.send_message(msg)
                smtp.quit()
                print(f"[EmailReader] Reply sent to {original_sender}")
                return True, f"Reply sent successfully to {original_sender}"
            except smtplib.SMTPAuthenticationError:
                return False, "SMTP authentication failed. Check GMAIL_USERNAME and GMAIL_PASSWORD."
            except Exception as exc:
                return False, f"SMTP error: {exc}"
        
        except Exception as exc:
            return False, f"Error composing reply: {exc}"

    def sync_unread(self) -> list[dict]:
        """Unread-only sync: fetch → thread → save. Returns processed messages."""
        msgs = self.fetch_unread()
        if not msgs:
            return []
        msgs = self.build_threads(msgs)
        self.save(msgs)
        print(f"[EmailReader] Saved {len(msgs)} unread email(s)")
        return msgs

    def send_new_email(self, to: str, subject: str, body: str) -> tuple[bool, str, str]:
        """
        Sends a brand-new email (not a reply to something already in the DB --
        use reply_to_email for that). Used for the "ask a human" fallback:
        the outbound question itself, not a reply to the client.

        Returns (success, message, message_id_of_sent_email). The message_id
        is what you thread pending_actions against, so the human's reply
        (which will carry In-Reply-To/References pointing back at it) can be
        matched up later.
        """
        user = os.getenv("GMAIL_USERNAME")
        app_pwd = os.getenv("GMAIL_PASSWORD")
        if not user or not app_pwd:
            return False, "GMAIL_USERNAME or GMAIL_PASSWORD not set in .env", ""

        msg = MIMEMultipart()
        msg["From"] = user
        msg["To"] = to
        msg["Subject"] = subject
        generated_id = email.utils.make_msgid()
        msg["Message-ID"] = generated_id
        msg.attach(MIMEText(body, "plain"))

        try:
            smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            smtp.login(user, app_pwd)
            smtp.send_message(msg)
            smtp.quit()
            print(f"[EmailReader] Question email sent to {to}")
            return True, f"Sent to {to}", generated_id
        except smtplib.SMTPAuthenticationError:
            return False, "SMTP authentication failed. Check GMAIL_USERNAME and GMAIL_PASSWORD.", ""
        except Exception as exc:
            return False, f"SMTP error: {exc}", ""

    @staticmethod
    def new_ref_token() -> str:
        """Short unique token embedded in outbound question subjects, e.g.
        '[REF-a83f2c91]'. Lets a reply be matched exactly even if the mail
        client stripped In-Reply-To/References -- no fuzzy subject guessing."""
        return uuid.uuid4().hex[:8]

    @staticmethod
    def embed_ref_token(subject: str, token: str) -> str:
        return f"{subject} [REF-{token}]"

    _REF_TOKEN_RE = re.compile(r"\[REF-([a-f0-9]{8})\]")

    @classmethod
    def extract_ref_token(cls, subject: str) -> Optional[str]:
        match = cls._REF_TOKEN_RE.search(subject or "")
        return match.group(1) if match else None

    def create_pending_action(self, question_message_id: str, action_type: str, payload_json: str, asked_to: str, question_subject: str = "", ref_token: str = ""):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO pending_actions (question_message_id, action_type, payload_json, asked_to, question_subject, ref_token, status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """, (question_message_id, action_type, payload_json, asked_to, question_subject, ref_token))
        conn.commit()
        conn.close()

    def find_pending_action_for_reply(self, msg: dict) -> Optional[tuple]:
        """
        Given a freshly-fetched message dict (from _parse_ids), checks whether
        it's a reply to a question we're waiting on.

        Tries two ways, in order:
        1. Header-based: its In-Reply-To or any References entry matches a
           pending_actions.question_message_id. Reliable, works whenever the
           reply preserves standard email headers.
        2. Ref-token fallback: some mail clients (notably some mobile apps)
           strip In-Reply-To/References on reply. Every outbound question's
           subject carries a unique "[REF-xxxxxxxx]" tag; if headers don't
           match, this extracts that tag from the reply's subject (quoted
           replies normally keep the original subject line intact even when
           headers are dropped) and looks it up exactly -- no ambiguity,
           unlike guessing from subject text alone.

        Returns the pending_actions row (as a tuple) if matched, else None.
        """
        candidates = [msg.get("in_reply_to", "")] + (msg.get("references") or [])
        candidates = [c for c in candidates if c]

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()

        row = None
        if candidates:
            placeholders = ",".join("?" for _ in candidates)
            cur.execute(f"""
            SELECT question_message_id, action_type, payload_json, asked_to, status
            FROM pending_actions
            WHERE question_message_id IN ({placeholders}) AND status = 'pending'
            """, candidates)
            row = cur.fetchone()

        if not row:
            token = self.extract_ref_token(msg.get("subject", ""))
            if token:
                cur.execute("""
                SELECT question_message_id, action_type, payload_json, asked_to, status
                FROM pending_actions
                WHERE status = 'pending' AND ref_token = ?
                """, (token,))
                row = cur.fetchone()
                if row:
                    print(f"[EmailReader] Matched reply to pending action '{row[0]}' by ref token (no reply headers present)")

        conn.close()
        return row



    def resolve_pending_action(self, question_message_id: str, status: str):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute("""
        UPDATE pending_actions SET status = ?, resolved_at = CURRENT_TIMESTAMP
        WHERE question_message_id = ?
        """, (status, question_message_id))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    reader = EmailReader()
    reader.connect()
    try:
        new = reader.sync_unread()
        if new:
            print("\n--- AGENT CONTEXT ---")
            print(reader.build_agent_context())
    finally:
        reader.disconnect()