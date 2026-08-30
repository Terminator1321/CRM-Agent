#!/usr/bin/env python3
"""
magma_cli.py

Command-line client for the MagmaAssistance FastAPI/uvicorn backend
(server.py). Talks to the same endpoints the web UI would, so you can
use/test the backend without touching the UI yet.

Usage:
    python magma_cli.py                       # interactive chat REPL
    python magma_cli.py --url http://localhost:8050
    python magma_cli.py chat "hello there"     # one-shot message
    python magma_cli.py health
    python magma_cli.py login <api_key> <api_secret>
    python magma_cli.py logout
    python magma_cli.py upload path/to/file.pdf
    python magma_cli.py audit sessions
    python magma_cli.py audit transcript <session_id>
    python magma_cli.py audit export [session_id]

Requires: pip install requests
"""

import argparse
import json
import os
import sys
from typing import Optional

try:
    import requests
except ImportError:
    print("This CLI needs the 'requests' package: pip install requests")
    sys.exit(1)

def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no python-dotenv dependency). Only sets
    variables that aren't already in the environment, so real env vars
    always win. Looks in the current directory by default."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()

DEFAULT_URL = os.environ.get("MAGMA_API_URL", "http://localhost:8050")
DEFAULT_SESSION = os.environ.get("MAGMA_SESSION_ID", "cli")
DEFAULT_CRM_API_KEY = (
    os.environ.get("CRM_API_KEY") or os.environ.get("CRM_API_KEY") or os.environ.get("FRAPPE_API_KEY")
)
DEFAULT_CRM_API_SECRET = (
    os.environ.get("CRM_API_SECRET") or os.environ.get("CRM_API_SECRET") or os.environ.get("FRAPPE_API_SECRET")
)


class MagmaClient:
    def __init__(self, base_url: str, session_id: str = DEFAULT_SESSION, user_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.user_id = user_id

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ---- core endpoints -------------------------------------------------

    def health(self) -> dict:
        r = requests.get(self._url("/api/health"), timeout=10)
        r.raise_for_status()
        return r.json()

    def chat(self, message: str) -> dict:
        payload = {"message": message, "session_id": self.session_id}
        if self.user_id:
            payload["user_id"] = self.user_id
        r = requests.post(self._url("/api/chat"), json=payload, timeout=120)
        if not r.ok:
            self._raise_with_detail(r)
        return r.json()

    def chat_stream(self, message: str):
        payload = {"message": message, "session_id": self.session_id}
        if self.user_id:
            payload["user_id"] = self.user_id
        with requests.post(self._url("/api/chat/stream"), json=payload, stream=True, timeout=120) as r:
            if not r.ok:
                self._raise_with_detail(r)
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    event = json.loads(payload_str)
                except Exception:
                    continue
                yield event

    def identify_session(self, api_key: str, api_secret: str) -> dict:
        payload = {
            "session_id": self.session_id,
            "crm_api_key": api_key,
            "crm_api_secret": api_secret,
        }
        r = requests.post(self._url("/api/session/identify"), json=payload, timeout=30)
        if not r.ok:
            self._raise_with_detail(r)
        return r.json()

    def logout_session(self) -> dict:
        r = requests.post(
            self._url("/api/session/logout"),
            data={"session_id": self.session_id},
            timeout=30,
        )
        if not r.ok:
            self._raise_with_detail(r)
        return r.json()

    def upload_document(self, filepath: str) -> dict:
        return self._upload("/api/upload-document", filepath)

    def _upload(self, path: str, filepath: str) -> dict:
        content_type = _guess_content_type(filepath)
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f, content_type)}
            data = {"session_id": self.session_id, "user_id": self.user_id or "cli-user"}
            r = requests.post(self._url(path), files=files, data=data, timeout=120)
        if not r.ok:
            self._raise_with_detail(r)
        return r.json()

    def audit_sessions(self, since: Optional[str] = None, limit: int = 100) -> dict:
        params = {"limit": limit}
        if since:
            params["since"] = since
        r = requests.get(self._url("/api/audit/sessions"), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def audit_transcript(self, session_id: str) -> dict:
        r = requests.get(self._url(f"/api/audit/sessions/{session_id}"), timeout=30)
        if not r.ok:
            self._raise_with_detail(r)
        return r.json()

    def audit_export(self, session_id: Optional[str] = None, out_path: Optional[str] = None) -> str:
        params = {}
        if session_id:
            params["session_id"] = session_id
        r = requests.get(self._url("/api/audit/export"), params=params, timeout=60)
        if not r.ok:
            self._raise_with_detail(r)
        default_name = f"audit_{session_id}.json" if session_id else "audit_all_sessions.json"
        out_path = out_path or default_name
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path

    def query_audio(self, filepath: str) -> dict:
        content_type = "audio/wav"
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f, content_type)}
            data = {"session_id": self.session_id, "user_id": self.user_id or "cli-user"}
            r = requests.post(self._url("/query"), files=files, data=data, timeout=120)
        if not r.ok:
            self._raise_with_detail(r)
        return r.json()

    @staticmethod
    def _raise_with_detail(r: "requests.Response"):
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {detail}")


def _guess_content_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")


# ---- pretty printing ------------------------------------------------------

def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _record_wav(seconds: float, samplerate: int = 16000) -> str:
    import sounddevice as sd
    import soundfile as sf
    import tempfile

    frames = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, frames, samplerate)
    return path


def _play_audio_bytes(audio_b64: str) -> None:
    import base64
    import tempfile

    audio_bytes = base64.b64decode(audio_b64)
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    try:
        from playsound import playsound
        playsound(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def run_voice_loop(client: MagmaClient, seconds: float = 5.0):
    try:
        import sounddevice  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        print("Voice mode needs: pip install sounddevice soundfile playsound")
        return

    print(f"Voice mode. Recording {seconds}s per turn. Press Ctrl+C to stop.\n")
    while True:
        try:
            input("Press Enter to speak...")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        wav_path = _record_wav(seconds)
        try:
            result = client.query_audio(wav_path)
        except Exception as e:
            print(f"[error] {e}\n")
            continue
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

        print(f"You said: {result.get('query', '')}")
        print(f"AI: {result.get('response', '')}\n")

        audio_b64 = result.get("audio")
        if audio_b64:
            try:
                _play_audio_bytes(audio_b64)
            except Exception as e:
                print(f"[warning] could not play audio reply: {e}\n")


# ---- interactive REPL ------------------------------------------------------

REPL_HELP = """\
Commands:
  <message>                 send a chat message
  /session [name]           show or switch session id (current: {session})
  /login [key] [secret]     bind Frappe CRM credentials (defaults to .env's CRM_API_KEY/SECRET)
  /voice [seconds]          start a mic-based voice conversation loop (fixed-duration)
  /logout                   unbind credentials from this session
  /upload <path>            upload a general document (PDF/JPEG/PNG)
  /health                   check backend health
  /audit sessions           list audit sessions
  /audit transcript <id>    show transcript for a session
  /audit export [id]        export audit log to a local JSON file
  /help                     show this help
  /quit                     exit

Run `python magma_voice.py` separately for realtime full-duplex voice with barge-in.
"""


def run_repl(client: MagmaClient):
    print(f"Connected to {client.base_url}  (session: {client.session_id})")
    try:
        health = client.health()
        print(f"Backend status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"[warning] could not reach backend health endpoint: {e}")
    print("Type a message to chat, or /help for commands. /quit to exit.\n")

    while True:
        try:
            line = input(f"[{client.session_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        if line.startswith("/"):
            if not _handle_command(client, line):
                break
            continue

        try:
            line_open = False
            for event in client.chat_stream(line):
                etype = event.get("type")
                if etype == "token":
                    sys.stdout.write(event.get("text", ""))
                    sys.stdout.flush()
                    line_open = True
                elif etype == "tool_call":
                    if line_open:
                        print()
                        line_open = False
                    print(f"[calling {event.get('name')}({event.get('args')})]")
                elif etype == "tool_result":
                    print(f"[{event.get('name')} -> {event.get('result')}]")
                elif etype == "error":
                    if line_open:
                        print()
                        line_open = False
                    print(f"[error] {event.get('message')}")
                elif etype == "done":
                    if line_open:
                        print()
                    print()
        except Exception as e:
            print(f"[error] {e}\n")


def _handle_command(client: MagmaClient, line: str) -> bool:
    """Returns False if the REPL should exit."""
    parts = line.split()
    cmd = parts[0].lower()

    if cmd in ("/quit", "/exit"):
        print("Bye.")
        return False

    if cmd == "/help":
        print(REPL_HELP.format(session=client.session_id))

    elif cmd == "/session":
        if len(parts) >= 2:
            client.session_id = parts[1]
            print(f"Switched to session '{client.session_id}'")
        else:
            print(f"Current session: {client.session_id}")

    elif cmd == "/login":
        if len(parts) == 3:
            api_key, api_secret = parts[1], parts[2]
        elif len(parts) == 1 and DEFAULT_CRM_API_KEY and DEFAULT_CRM_API_SECRET:
            api_key, api_secret = DEFAULT_CRM_API_KEY, DEFAULT_CRM_API_SECRET
            print("(using CRM_API_KEY / CRM_API_SECRET from .env)")
        else:
            print("Usage: /login <api_key> <api_secret>  (or /login with no args to use .env)")
            return True
        try:
            result = client.identify_session(api_key, api_secret)
            print(f"Authenticated as {result.get('user')} (roles: {result.get('roles')})")
        except Exception as e:
            print(f"[error] {e}")

    elif cmd == "/voice":
        seconds = float(parts[1]) if len(parts) >= 2 else 5.0
        run_voice_loop(client, seconds)

    elif cmd == "/logout":
        try:
            client.logout_session()
            print("Logged out of current session.")
        except Exception as e:
            print(f"[error] {e}")

    elif cmd == "/upload":
        if len(parts) < 2:
            print("Usage: /upload <path>")
        else:
            try:
                result = client.upload_document(" ".join(parts[1:]))
                print(f"{result.get('status')}: {result.get('message')}")
            except Exception as e:
                print(f"[error] {e}")

    elif cmd == "/uploadpo":
        if len(parts) < 2:
            print("Usage: /uploadpo <path>")
        else:
            try:
                result = client.upload_po(" ".join(parts[1:]))
                print(f"{result.get('status')}: {result.get('message')}")
            except Exception as e:
                print(f"[error] {e}")

    elif cmd == "/health":
        try:
            _print_json(client.health())
        except Exception as e:
            print(f"[error] {e}")

    elif cmd == "/audit":
        if len(parts) < 2:
            print("Usage: /audit sessions | /audit transcript <id> | /audit export [id]")
        elif parts[1] == "sessions":
            try:
                _print_json(client.audit_sessions())
            except Exception as e:
                print(f"[error] {e}")
        elif parts[1] == "transcript" and len(parts) >= 3:
            try:
                _print_json(client.audit_transcript(parts[2]))
            except Exception as e:
                print(f"[error] {e}")
        elif parts[1] == "export":
            sid = parts[2] if len(parts) >= 3 else None
            try:
                path = client.audit_export(sid)
                print(f"Saved to {path}")
            except Exception as e:
                print(f"[error] {e}")
        else:
            print("Usage: /audit sessions | /audit transcript <id> | /audit export [id]")

    else:
        print(f"Unknown command: {cmd}. Type /help for a list.")

    return True


# ---- argparse subcommands (for one-shot / scripted use) --------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI client for the MagmaAssistance backend.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Backend base URL (default: {DEFAULT_URL})")
    parser.add_argument("--session", default=DEFAULT_SESSION, help=f"Session id (default: {DEFAULT_SESSION})")
    parser.add_argument("--user", default=None, help="Optional user id to attach to requests")

    sub = parser.add_subparsers(dest="command")

    p_chat = sub.add_parser("chat", help="Send a single chat message and print the reply")
    p_chat.add_argument("message", nargs="+", help="Message text")

    sub.add_parser("health", help="Check backend health")

    p_login = sub.add_parser("login", help="Bind Frappe CRM credentials to the session (defaults to .env's CRM_API_KEY/SECRET)")
    p_login.add_argument("api_key", nargs="?", default=DEFAULT_CRM_API_KEY)
    p_login.add_argument("api_secret", nargs="?", default=DEFAULT_CRM_API_SECRET)

    sub.add_parser("logout", help="Unbind credentials from the session")

    p_upload = sub.add_parser("upload", help="Upload a general document")
    p_upload.add_argument("path")


    p_voice = sub.add_parser("voice", help="Mic-based voice conversation loop")
    p_voice.add_argument("--seconds", type=float, default=5.0)

    p_audit = sub.add_parser("audit", help="Audit log operations")
    audit_sub = p_audit.add_subparsers(dest="audit_command")
    audit_sub.add_parser("sessions", help="List audit sessions")
    p_transcript = audit_sub.add_parser("transcript", help="Show transcript for a session")
    p_transcript.add_argument("session_id")
    p_export = audit_sub.add_parser("export", help="Export audit log to a JSON file")
    p_export.add_argument("session_id", nargs="?", default=None)
    p_export.add_argument("--out", default=None, help="Output file path")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    client = MagmaClient(args.url, session_id=args.session, user_id=args.user)

    if args.command is None:
        run_repl(client)
        return

    try:
        if args.command == "chat":
            result = client.chat(" ".join(args.message))
            print(result.get("reply", ""))

        elif args.command == "health":
            _print_json(client.health())

        elif args.command == "login":
            if not args.api_key or not args.api_secret:
                print("[error] No api_key/api_secret given and none found in .env (CRM_API_KEY/CRM_API_SECRET).")
                sys.exit(1)
            _print_json(client.identify_session(args.api_key, args.api_secret))

        elif args.command == "logout":
            _print_json(client.logout_session())

        elif args.command == "upload":
            _print_json(client.upload_document(args.path))

        elif args.command == "voice":
            run_voice_loop(client, args.seconds)

        elif args.command == "audit":
            if args.audit_command == "sessions":
                _print_json(client.audit_sessions())
            elif args.audit_command == "transcript":
                _print_json(client.audit_transcript(args.session_id))
            elif args.audit_command == "export":
                path = client.audit_export(args.session_id, args.out)
                print(f"Saved to {path}")
            else:
                parser.parse_args(["audit", "--help"])
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()