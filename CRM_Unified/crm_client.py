"""Thin, standalone client for a separate Frappe CRM instance."""
import contextlib
import contextvars
import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_TIMEOUT = int(os.getenv("CRM_REQUEST_TIMEOUT", "20"))
META_TTL = int(os.getenv("CRM_METADATA_TTL", "900"))

class CRMIdentity:
    __slots__ = ("api_key", "api_secret", "user", "roles")
    def __init__(self, api_key: str, api_secret: str, user: str = None, roles=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.user = user
        self.roles = roles or []

_active_identity: contextvars.ContextVar[Optional[CRMIdentity]] = contextvars.ContextVar("crm_identity", default=None)

@contextlib.contextmanager
def use_identity(identity: Optional[CRMIdentity]):
    token = _active_identity.set(identity)
    try:
        yield
    finally:
        _active_identity.reset(token)

def current_identity() -> Optional[CRMIdentity]:
    return _active_identity.get()

class CRMClient:
    def __init__(self):
        self.base_url = (os.getenv("CRM_URL") or os.getenv("FRAPPE_CRM_URL") or "").rstrip("/")
        self.api_key = os.getenv("CRM_API_KEY")
        self.api_secret = os.getenv("CRM_API_SECRET")
        self.username = os.getenv("CRM_USERNAME")
        self.password = os.getenv("CRM_PASSWORD")
        self.session = requests.Session()
        self._logged_in = False
        self._meta_cache = {}

    def _headers(self):
        identity = current_identity()
        key = identity.api_key if identity else self.api_key
        secret = identity.api_secret if identity else self.api_secret
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if key and secret:
            headers["Authorization"] = f"token {key}:{secret}"
        return headers

    def _ensure_session_login(self):
        if self._logged_in or not (self.username and self.password):
            return
        response = self.session.post(
            self._url("/api/method/login"),
            data={"usr": self.username, "pwd": self.password},
            headers={"Accept": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        if not response.ok:
            raise PermissionError("CRM session login failed. Check CRM_USERNAME/CRM_PASSWORD.")
        self._logged_in = True

    def _url(self, path):
        if not self.base_url:
            raise RuntimeError("CRM_URL is not configured.")
        return f"{self.base_url}{path}"

    def _request(self, method, path, *, params=None, json_body=None, action="access CRM"):
        if not (self.api_key and self.api_secret) and not current_identity():
            self._ensure_session_login()
        response = self.session.request(method, self._url(path), headers=self._headers(), params=params,
                                        json=json_body, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 401:
            raise PermissionError("CRM authentication failed. Check CRM_API_KEY/CRM_API_SECRET.")
        if response.status_code == 403:
            raise PermissionError(f"CRM permission denied while trying to {action}.")
        if not response.ok:
            try:
                detail = response.json().get("message") or response.text
            except Exception:
                detail = response.text
            raise RuntimeError(f"CRM request failed ({response.status_code}): {detail}")
        try:
            payload = response.json()
        except ValueError:
            return response.text
        return payload.get("message", payload)

    def get_logged_user(self):
        return self._request("GET", "/api/method/frappe.auth.get_logged_user", action="identify CRM user")

    def resolve_identity(self, api_key, api_secret):
        old = self.api_key, self.api_secret
        self.api_key, self.api_secret = api_key, api_secret
        try:
            user = self.get_logged_user()
            if isinstance(user, dict):
                user = user.get("user") or user.get("message")
            if not user or user == "Guest":
                raise PermissionError("CRM credentials did not resolve to an authenticated user.")
            roles = []
            try:
                user_doc = self.get_doc("User", user)
                roles = [r.get("role") for r in user_doc.get("roles", []) if r.get("role")] if isinstance(user_doc, dict) else []
            except Exception:
                pass
            return CRMIdentity(api_key, api_secret, user, roles)
        finally:
            self.api_key, self.api_secret = old

    def get_list(self, doctype, fields=None, filters=None, order_by=None, limit=20, start=0, or_filters=None):
        params = {"limit_page_length": max(1, min(int(limit or 20), 100))}
        if start: params["limit_start"] = int(start)
        if fields: params["fields"] = json.dumps(fields)
        if filters: params["filters"] = json.dumps(filters)
        if or_filters: params["or_filters"] = json.dumps(or_filters)
        if order_by: params["order_by"] = order_by
        return self._request("GET", f"/api/resource/{quote(doctype, safe='')}", params=params, action=f"list {doctype}")

    def get_doc(self, doctype, name):
        return self._request("GET", f"/api/resource/{quote(doctype, safe='')}/{quote(str(name), safe='')}", action=f"read {doctype}")

    def create_doc(self, doctype, data):
        return self._request("POST", f"/api/resource/{quote(doctype, safe='')}", json_body={"doctype": doctype, **data}, action=f"create {doctype}")

    def update_doc(self, doctype, name, data):
        return self._request("PUT", f"/api/resource/{quote(doctype, safe='')}/{quote(str(name), safe='')}", json_body=data, action=f"update {doctype}")

    def delete_doc(self, doctype, name):
        return self._request("DELETE", f"/api/resource/{quote(doctype, safe='')}/{quote(str(name), safe='')}", action=f"delete {doctype}")

    def call_method(self, method, params=None, http_method="GET"):
        return self._request(http_method, f"/api/method/{method}", params=params if http_method == "GET" else None,
                             json_body=params if http_method != "GET" else None, action=f"call {method}")

    def get_fields(self, doctype):
        now = time.time()
        cached = self._meta_cache.get(doctype)
        if cached and now - cached[0] < META_TTL:
            return cached[1]
        fields = self.call_method("crm.api.doc.get_fields", {"doctype": doctype})
        normalized = []
        for f in fields or []:
            if hasattr(f, "as_dict"):
                f = f.as_dict()
            normalized.append({k: f.get(k) for k in ("fieldname", "label", "fieldtype", "options", "reqd", "read_only", "hidden") if f.get(k) is not None})
        self._meta_cache[doctype] = (now, normalized)
        return normalized

    def get_filterable_fields(self, doctype):
        return self.call_method("crm.api.doc.get_filterable_fields", {"doctype": doctype})

crm_client = CRMClient()
