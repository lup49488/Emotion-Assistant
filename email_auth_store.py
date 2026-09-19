"""Email/password identities that preserve the existing internal user ID."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

import config
from auth_store import PBKDF2_ITERATIONS, migrate_legacy_auth
from email_delivery import configured as email_delivery_configured
from email_delivery import send_verification_code
from session_store import validate_user_id
from sqlite_store import connection, ensure_user, sqlite_enabled
from turnstile import verify_email_auth_token


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_HASHER = PasswordHasher(memory_cost=19_456, time_cost=2, parallelism=1)


class EmailAuthError(ValueError):
    pass


class EmailAuthUnavailable(EmailAuthError):
    pass


@dataclass(frozen=True)
class VerifiedIntent:
    challenge_id: str
    email: str
    purpose: str
    expires_at: int


def normalize_email(email: str) -> str:
    value = (email or "").strip().casefold()
    if len(value) > 254 or not EMAIL_PATTERN.fullmatch(value):
        raise EmailAuthError("Enter a valid email address.")
    return value


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


class EmailAuthService:
    def __init__(
        self,
        intent_secret: bytes,
        *,
        send_code: Callable[[str, str], None] = send_verification_code,
        validate_turnstile: Callable[[str, str], bool] = verify_email_auth_token,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._intent_secret = intent_secret
        self._send_code = send_code
        self._validate_turnstile = validate_turnstile
        self._clock = clock

    def enabled(self) -> bool:
        turnstile_ready = not config.EMAIL_AUTH_TURNSTILE_REQUIRED or bool(
            config.TURNSTILE_SITE_KEY and config.TURNSTILE_SECRET_KEY
        )
        return config.EMAIL_AUTH_ENABLED and sqlite_enabled() and email_delivery_configured() and turnstile_ready

    def public_config(self) -> dict[str, object]:
        return {
            "email_auth_enabled": self.enabled(),
            "legacy_login_enabled": config.EMAIL_AUTH_LEGACY_LOGIN_ENABLED,
            "turnstile_required": config.EMAIL_AUTH_TURNSTILE_REQUIRED,
            "turnstile_site_key": config.TURNSTILE_SITE_KEY if self.enabled() else "",
            "turnstile_action": config.TURNSTILE_EMAIL_AUTH_ACTION,
        }

    def legacy_login_enabled(self) -> bool:
        return config.EMAIL_AUTH_LEGACY_LOGIN_ENABLED

    def _require_enabled(self) -> None:
        if not self.enabled():
            raise EmailAuthUnavailable("Email sign-in is not available yet.")

    def _hash_code(self, challenge_id: str, code: str) -> str:
        return hmac.new(
            self._intent_secret,
            f"email-code:{challenge_id}:{code}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _issue_intent(self, challenge_id: str, email: str, purpose: str, expires_at: int) -> str:
        payload = {"cid": challenge_id, "email": email, "purpose": purpose, "exp": expires_at}
        encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).decode("ascii").rstrip("=")
        signature = hmac.new(self._intent_secret, f"email-intent:{encoded}".encode("ascii"), hashlib.sha256).digest()
        return f"{encoded}.{base64.urlsafe_b64encode(signature).decode('ascii').rstrip('=')}"

    def _read_intent(self, value: str, expected_purpose: str) -> VerifiedIntent:
        encoded, separator, supplied_signature = (value or "").partition(".")
        if not separator:
            raise EmailAuthError("Verification has expired. Request a new code.")
        expected_signature = hmac.new(self._intent_secret, f"email-intent:{encoded}".encode("ascii"), hashlib.sha256).digest()
        try:
            signature = base64.urlsafe_b64decode(supplied_signature + "=" * (-len(supplied_signature) % 4))
            payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            raise EmailAuthError("Verification has expired. Request a new code.") from None
        if not hmac.compare_digest(expected_signature, signature):
            raise EmailAuthError("Verification has expired. Request a new code.")
        try:
            intent = VerifiedIntent(str(payload["cid"]), normalize_email(str(payload["email"])), str(payload["purpose"]), int(payload["exp"]))
        except (KeyError, TypeError, ValueError, EmailAuthError):
            raise EmailAuthError("Verification has expired. Request a new code.") from None
        if intent.purpose != expected_purpose or intent.expires_at <= int(self._clock()):
            raise EmailAuthError("Verification has expired. Request a new code.")
        return intent

    def _audit(self, conn: sqlite3.Connection, user_id: str | None, event_type: str, outcome: str, request_id: str) -> None:
        conn.execute(
            "INSERT INTO email_auth_audit_events(user_id, event_type, outcome, request_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, event_type, outcome, request_id[:64], _now_text()),
        )

    def start_challenge(self, email: str, purpose: str, turnstile_token: str, client_ip: str, request_id: str) -> str:
        self._require_enabled()
        normalized = normalize_email(email)
        if purpose not in {"registration", "legacy_migration"}:
            raise EmailAuthError("Invalid verification request.")
        if not self._validate_turnstile(turnstile_token, client_ip):
            raise EmailAuthError("Verification could not be completed. Please try again.")
        now = int(self._clock())
        with connection() as conn:
            recent = conn.execute(
                "SELECT COUNT(*) FROM email_challenges WHERE email_normalized = ? AND created_at >= ?",
                (normalized, now - config.EMAIL_AUTH_SEND_WINDOW_SECONDS),
            ).fetchone()[0]
            if recent >= config.EMAIL_AUTH_MAX_SENDS_PER_WINDOW:
                self._audit(conn, None, "challenge_send", "rate_limited", request_id)
                raise EmailAuthError("Too many verification attempts. Please try again later.")
            conn.execute(
                "UPDATE email_challenges SET consumed_at = ? WHERE email_normalized = ? AND consumed_at IS NULL AND verified_at IS NULL",
                (now, normalized),
            )
            challenge_id = secrets.token_urlsafe(24)
            code = f"{secrets.randbelow(100_000_000):08d}"
            conn.execute(
                "INSERT INTO email_challenges(id, email_normalized, purpose, code_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (challenge_id, normalized, purpose, self._hash_code(challenge_id, code), now + config.EMAIL_AUTH_CODE_TTL_SECONDS, now),
            )
            self._audit(conn, None, "challenge_send", "created", request_id)
        try:
            self._send_code(normalized, code)
        except Exception as exc:
            with connection() as conn:
                conn.execute("DELETE FROM email_challenges WHERE id = ? AND verified_at IS NULL", (challenge_id,))
                self._audit(conn, None, "challenge_send", "delivery_failed", request_id)
            raise EmailAuthError("We could not send a verification code. Please try again.") from exc
        return challenge_id

    def verify_challenge(self, challenge_id: str, code: str, purpose: str, request_id: str) -> str:
        self._require_enabled()
        now = int(self._clock())
        with connection() as conn:
            row = conn.execute(
                "SELECT email_normalized, purpose, code_hash, expires_at, attempts, verified_at, consumed_at FROM email_challenges WHERE id = ?",
                (challenge_id,),
            ).fetchone()
            if row is None or row["purpose"] != purpose or row["consumed_at"] is not None or row["expires_at"] <= now:
                self._audit(conn, None, "challenge_verify", "rejected", request_id)
                raise EmailAuthError("Verification could not be completed. Request a new code.")
            if row["attempts"] >= config.EMAIL_AUTH_MAX_CODE_ATTEMPTS:
                self._audit(conn, None, "challenge_verify", "attempt_limit", request_id)
                raise EmailAuthError("Verification could not be completed. Request a new code.")
            expected = self._hash_code(challenge_id, (code or "").strip())
            if not hmac.compare_digest(expected, row["code_hash"]):
                conn.execute("UPDATE email_challenges SET attempts = attempts + 1 WHERE id = ?", (challenge_id,))
                self._audit(conn, None, "challenge_verify", "rejected", request_id)
                raise EmailAuthError("Verification could not be completed. Request a new code.")
            if row["verified_at"] is None:
                conn.execute("UPDATE email_challenges SET verified_at = ? WHERE id = ?", (now, challenge_id))
            self._audit(conn, None, "challenge_verify", "verified", request_id)
            return self._issue_intent(challenge_id, row["email_normalized"], purpose, min(row["expires_at"], now + 300))

    def _require_challenge(self, conn: sqlite3.Connection, intent: VerifiedIntent) -> None:
        row = conn.execute(
            "SELECT email_normalized, purpose, expires_at, verified_at, consumed_at FROM email_challenges WHERE id = ?",
            (intent.challenge_id,),
        ).fetchone()
        if row is None or row["email_normalized"] != intent.email or row["purpose"] != intent.purpose or row["verified_at"] is None or row["consumed_at"] is not None or row["expires_at"] <= int(self._clock()):
            raise EmailAuthError("Verification has expired. Request a new code.")

    def _new_password_hash(self, password: str) -> str:
        if len(password or "") < 12 or len(password) > 512:
            raise EmailAuthError("Use a password between 12 and 512 characters.")
        return PASSWORD_HASHER.hash(password)

    def register(self, intent_token: str, password: str, request_id: str) -> str:
        self._require_enabled()
        intent = self._read_intent(intent_token, "registration")
        password_hash = self._new_password_hash(password)
        now = int(self._clock())
        with connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_challenge(conn, intent)
            user_id = f"usr_{secrets.token_urlsafe(18)}"
            ensure_user(conn, user_id)
            try:
                conn.execute(
                    "INSERT INTO email_identities(user_id, email_normalized, email_verified_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, intent.email, _now_text(), _now_text(), _now_text()),
                )
            except sqlite3.IntegrityError as exc:
                self._audit(conn, None, "registration", "email_already_bound", request_id)
                raise EmailAuthError("We could not create that account. Try signing in or account recovery.") from exc
            version = secrets.token_urlsafe(18)
            conn.execute(
                "INSERT INTO password_credentials(user_id, password_hash, changed_at, credential_version) VALUES (?, ?, ?, ?)",
                (user_id, password_hash, _now_text(), version),
            )
            conn.execute("UPDATE email_challenges SET consumed_at = ? WHERE id = ?", (now, intent.challenge_id))
            self._audit(conn, user_id, "registration", "succeeded", request_id)
            return user_id

    def migrate_legacy(self, intent_token: str, legacy_user_id: str, access_key: str, password: str, request_id: str) -> str:
        self._require_enabled()
        intent = self._read_intent(intent_token, "legacy_migration")
        password_hash = self._new_password_hash(password)
        try:
            user_id = validate_user_id((legacy_user_id or "").strip())
            migrate_legacy_auth(user_id)
        except (ValueError, EmailAuthError):
            raise EmailAuthError("We could not complete migration with those details.") from None
        now = int(self._clock())
        with connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_challenge(conn, intent)
            legacy = conn.execute("SELECT salt, password_hash FROM auth_credentials WHERE user_id = ?", (user_id,)).fetchone()
            if legacy is None:
                self._audit(conn, None, "legacy_migration", "rejected", request_id)
                raise EmailAuthError("We could not complete migration with those details.")
            actual = hashlib.pbkdf2_hmac("sha256", (access_key or "").strip().encode("utf-8"), bytes.fromhex(legacy["salt"]), PBKDF2_ITERATIONS).hex()
            if not hmac.compare_digest(actual, legacy["password_hash"]):
                self._audit(conn, user_id, "legacy_migration", "rejected", request_id)
                raise EmailAuthError("We could not complete migration with those details.")
            try:
                conn.execute(
                    "INSERT INTO email_identities(user_id, email_normalized, email_verified_at, legacy_migrated_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, intent.email, _now_text(), _now_text(), _now_text(), _now_text()),
                )
            except sqlite3.IntegrityError as exc:
                self._audit(conn, user_id, "legacy_migration", "email_already_bound", request_id)
                raise EmailAuthError("We could not complete migration with those details.") from exc
            version = secrets.token_urlsafe(18)
            conn.execute(
                "INSERT INTO password_credentials(user_id, password_hash, changed_at, credential_version) VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET password_hash = excluded.password_hash, changed_at = excluded.changed_at, credential_version = excluded.credential_version",
                (user_id, password_hash, _now_text(), version),
            )
            conn.execute("UPDATE email_challenges SET consumed_at = ? WHERE id = ?", (now, intent.challenge_id))
            self._audit(conn, user_id, "legacy_migration", "succeeded", request_id)
            return user_id

    def login_password(self, email: str, password: str, request_id: str) -> str:
        self._require_enabled()
        normalized = normalize_email(email)
        with connection() as conn:
            row = conn.execute(
                "SELECT identity.user_id, credential.password_hash FROM email_identities AS identity JOIN password_credentials AS credential ON credential.user_id = identity.user_id WHERE identity.email_normalized = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                self._audit(conn, None, "password_login", "rejected", request_id)
                raise EmailAuthError("Email or password is incorrect.")
            try:
                valid = PASSWORD_HASHER.verify(row["password_hash"], password or "")
            except (InvalidHashError, VerificationError, VerifyMismatchError):
                valid = False
            if not valid:
                self._audit(conn, row["user_id"], "password_login", "rejected", request_id)
                raise EmailAuthError("Email or password is incorrect.")
            self._audit(conn, row["user_id"], "password_login", "succeeded", request_id)
            return str(row["user_id"])

    def credential_version(self, user_id: str) -> str | None:
        if not sqlite_enabled():
            return None
        with connection() as conn:
            row = conn.execute("SELECT credential_version FROM password_credentials WHERE user_id = ?", (user_id,)).fetchone()
        return str(row["credential_version"]) if row else None
