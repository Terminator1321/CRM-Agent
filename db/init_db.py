"""
db/init_db.py

Applies db/schema.sql against Postgres, using separate connection
settings (PGHOST / PGPORT / PGUSER / PGPASSWORD / PGDATABASE) from .env
-- no combined connection-string URL to build or escape.

Usage:
    python db/init_db.py

Requires:
    pip install psycopg2-binary python-dotenv
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("db-init")

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection_params() -> dict:
    """Reads PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE from the
    environment. Raises with a clear message listing exactly which
    variable is missing, rather than failing deep inside psycopg2."""
    required = {
        "host": "PGHOST",
        "port": "PGPORT",
        "user": "PGUSER",
        "password": "PGPASSWORD",
        "dbname": "PGDATABASE",
    }
    params = {}
    missing = []
    for key, env_name in required.items():
        value = os.getenv(env_name)
        if not value:
            missing.append(env_name)
        params[key] = value

    if missing:
        raise RuntimeError(
            "Missing Postgres connection settings in .env: "
            + ", ".join(missing)
            + "\nExpected all of: PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE"
        )

    return params


def apply_schema(connection_params: Optional[dict] = None) -> None:
    """Applies schema.sql using `connection_params` (or PG* env vars if
    not given). Idempotent -- schema.sql only uses CREATE TABLE/INDEX IF
    NOT EXISTS and DO-block guarded CREATE TYPE, so this is safe to call
    on every server startup, not just once by hand.
    """
    params = connection_params or get_connection_params()
    sql = SCHEMA_PATH.read_text(encoding="utf-8")

    conn = psycopg2.connect(**params)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info("Postgres schema applied/verified: sessions, audit_log, file_uploads.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    """CLI entry point: `python db/init_db.py`. Still works standalone if
    you want to apply the schema without starting the whole server."""
    logging.basicConfig(level=logging.INFO)
    try:
        apply_schema()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print("Schema applied successfully.")


if __name__ == "__main__":
    main()
