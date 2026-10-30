"""
Sign-in tokens for the web API.

A token is the user's role, contact and name, plus an expiry time, signed with
HMAC-SHA256 (standard library only). The browser keeps it and sends it as
`Authorization: Bearer <token>`. Nothing is stored on the server, so tokens
survive restarts; changing APP_SECRET signs everyone out.

There are no passwords for farmers and providers, the same as the Streamlit
app: they are identified by a validated phone number or email. Only the
administrator has a password.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException

from . import settings

TOKEN_DAYS = 7
ROLES = ("farmer", "provider", "admin")


@dataclass
class User:
    role: str
    contact: str
    name: str


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(user: User, now: float | None = None) -> str:
    body = {"role": user.role, "contact": user.contact, "name": user.name,
            "exp": int((now or time.time()) + TOKEN_DAYS * 86400)}
    payload = _b64(json.dumps(body, separators=(",", ":")).encode())
    sig = _b64(hmac.new(settings.app_secret(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def read_token(token: str, now: float | None = None) -> User | None:
    try:
        payload, sig = token.split(".")
        good = _b64(hmac.new(settings.app_secret(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        body = json.loads(_unb64(payload))
        if body["exp"] < (now or time.time()) or body["role"] not in ROLES:
            return None
        return User(body["role"], body["contact"], body.get("name", ""))
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def current_user(authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Please sign in.")
    user = read_token(authorization.split(" ", 1)[1].strip())
    if user is None:
        raise HTTPException(401, "Your sign-in has expired. Please sign in again.")
    return user


def require(role: str):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role != role:
            raise HTTPException(403, "This page is not available for your account.")
        return user
    return dep
