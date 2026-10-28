"""Small transactional application store; vectors remain in PostgreSQL."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from stacklens.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, owner TEXT NOT NULL REFERENCES users(id), filename TEXT NOT NULL, path TEXT NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', error TEXT, attempts INTEGER NOT NULL DEFAULT 0, chunks INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, lease_until REAL, lease_token TEXT, published REAL, UNIQUE(owner,digest));
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner,created);
"""


@contextmanager
def connection():
    path = get_settings().app_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    try:
        with db:
            yield db
    finally:
        db.close()


def password_hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 310_000).hex()


def create_user(username: str, password: str) -> dict:
    salt = secrets.token_hex(16)
    user = {"id": uuid4().hex, "username": username.strip().lower()}
    try:
        with connection() as db:
            db.execute(
                "INSERT INTO users VALUES(?,?,?,?)",
                (user["id"], user["username"], salt, password_hash(password, salt)),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username is unavailable.") from exc
    return user


def authenticate(username: str, password: str) -> dict | None:
    with connection() as db:
        row = db.execute(
            "SELECT * FROM users WHERE username=?", (username.strip().lower(),)
        ).fetchone()
    # Keep the expensive hash on the unknown-user path as well.
    actual = password_hash(password, row["salt"] if row else "00" * 16)
    if not row or not hmac.compare_digest(actual, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


def issue_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with connection() as db:
        db.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        db.execute(
            "INSERT INTO sessions VALUES(?,?,?)",
            (hashlib.sha256(token.encode()).hexdigest(), user_id, time.time() + 86400 * 7),
        )
    return token


def get_user(token: str | None) -> dict | None:
    if not token:
        return None
    with connection() as db:
        row = db.execute(
            "SELECT users.id,users.username FROM sessions JOIN users ON users.id=sessions.user_id WHERE token_hash=? AND expires>?",
            (hashlib.sha256(token.encode()).hexdigest(), time.time()),
        ).fetchone()
    return dict(row) if row else None


def revoke_session(token: str):
    with connection() as db:
        db.execute(
            "DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),)
        )


def new_job(owner: str, filename: str, path: Path, digest: str, job_id: str) -> tuple[dict, bool]:
    with connection() as db:
        db.execute(
            "INSERT OR IGNORE INTO jobs(id,owner,filename,path,digest,created) VALUES(?,?,?,?,?,?)",
            (job_id, owner, filename, str(path), digest, time.time()),
        )
        row = db.execute(
            "SELECT * FROM jobs WHERE owner=? AND digest=?", (owner, digest)
        ).fetchone()
    return dict(row), row["id"] == job_id


def get_job(job_id: str, owner: str | None = None) -> dict | None:
    with connection() as db:
        row = db.execute(
            "SELECT * FROM jobs WHERE id=?" + (" AND owner=?" if owner else ""),
            (job_id, owner) if owner else (job_id,),
        ).fetchone()
    return dict(row) if row else None


def list_jobs(owner: str) -> list[dict]:
    with connection() as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT * FROM jobs WHERE owner=? ORDER BY created DESC LIMIT 100", (owner,)
            )
        ]


def claim_job(job_id: str) -> dict | None:
    now = time.time()
    token = uuid4().hex
    with connection() as db:
        result = db.execute(
            "UPDATE jobs SET status='processing',attempts=attempts+1,lease_until=?,lease_token=?,error=NULL WHERE id=? AND (status='queued' OR (status='processing' AND lease_until<?))",
            (now + 900, token, job_id, now),
        )
        if result.rowcount != 1:
            return None
        return dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def finish_job(job: dict, *, chunks: int = 0, error: str | None = None):
    with connection() as db:
        db.execute(
            "UPDATE jobs SET status=?,chunks=?,error=?,lease_until=NULL WHERE id=? AND lease_token=?",
            ("failed" if error else "completed", chunks, error, job["id"], job["lease_token"]),
        )


def retry_job(job_id: str, owner: str) -> bool:
    with connection() as db:
        result = db.execute(
            "UPDATE jobs SET status='queued',error=NULL,published=NULL WHERE id=? AND owner=? AND status='failed'",
            (job_id, owner),
        )
        return result.rowcount == 1


def delete_job(job_id: str, owner: str) -> tuple[dict | None, bool]:
    """Tombstone a job so queued or in-flight workers cannot publish it again."""
    with connection() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        if row is None:
            return None, False
        if row["status"] == "deleted":
            return dict(row), False
        db.execute(
            "UPDATE jobs SET status='deleted',lease_until=NULL,lease_token=NULL,error=NULL "
            "WHERE id=? AND owner=?",
            (job_id, owner),
        )
        return dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()), True


def job_is_active(job_id: str, lease_token: str) -> bool:
    with connection() as db:
        row = db.execute(
            "SELECT 1 FROM jobs WHERE id=? AND lease_token=? AND status='processing'",
            (job_id, lease_token),
        ).fetchone()
    return row is not None


def pending_jobs() -> list[dict]:
    with connection() as db:
        db.execute(
            "UPDATE jobs SET status='queued',published=NULL WHERE status='processing' AND lease_until<?",
            (time.time(),),
        )
        return [
            dict(r)
            for r in db.execute(
                "SELECT * FROM jobs WHERE status='queued' AND published IS NULL ORDER BY created LIMIT 20"
            )
        ]


def mark_published(job_id: str):
    with connection() as db:
        db.execute("UPDATE jobs SET published=? WHERE id=?", (time.time(), job_id))
