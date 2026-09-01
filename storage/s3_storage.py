"""
storage/s3_storage.py

Uploads user-submitted files (PO scans, general documents, audio) to S3
and returns the metadata needed to write a file_uploads row via
db.postgres_audit_log.record_file_upload().

Credentials come from environment variables (see .env.example):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN      (optional -- only for temporary/STS credentials)
    AWS_REGION
    S3_BUCKET_NAME

In production, prefer an IAM instance role / task role over static keys
and simply omit AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY -- boto3 will
pick up the role automatically and there's nothing to leak or rotate.

Requires:
    pip install boto3 python-dotenv
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("s3-storage")

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Falls back to saving uploads on the server's own disk whenever S3 isn't
# configured (or a configured bucket can't be reached) -- so uploads still
# work end-to-end without AWS credentials, at the cost of not being durable
# across redeploys. Set S3_BUCKET_NAME + AWS credentials in .env to switch
# back to S3 with no other code changes.
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", "local_uploads")

_s3_client = None
_warned_local_fallback = False


def _s3_configured() -> bool:
    return bool(S3_BUCKET_NAME)


def _warn_local_fallback_once(reason: str):
    global _warned_local_fallback
    if not _warned_local_fallback:
        logger.warning(
            "S3 is not available (%s) -- falling back to storing uploaded "
            "files locally on the server at '%s'. Set S3_BUCKET_NAME + AWS "
            "credentials in .env for durable, off-server storage.",
            reason, os.path.abspath(LOCAL_STORAGE_DIR),
        )
        _warned_local_fallback = True


def _client():
    global _s3_client
    if _s3_client is None:
        if not S3_BUCKET_NAME:
            raise RuntimeError("S3_BUCKET_NAME is not set -- add it to .env.")
        # boto3 resolves credentials itself, in this order: explicit args
        # below -> env vars -> shared credentials file -> IAM role. We
        # pass explicit env vars only when present so an IAM role still
        # works if the vars are simply left unset.
        kwargs = {"region_name": AWS_REGION}
        if os.getenv("AWS_ACCESS_KEY_ID"):
            kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
            kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
            if os.getenv("AWS_SESSION_TOKEN"):
                kwargs["aws_session_token"] = os.getenv("AWS_SESSION_TOKEN")
        # Only used for local testing against LocalStack/MinIO -- leave
        # AWS_S3_ENDPOINT_URL unset in real environments so boto3 talks to
        # actual AWS.
        if os.getenv("AWS_S3_ENDPOINT_URL"):
            kwargs["endpoint_url"] = os.getenv("AWS_S3_ENDPOINT_URL")
        _s3_client = boto3.client("s3", **kwargs)
    return _s3_client


def _build_key(upload_kind: str, session_id: Optional[str], filename: str) -> str:
    """e.g. purchase_order/2026/07/26/<session>/<uuid>_invoice.pdf --
    date-partitioned so lifecycle rules / Athena partitioning work
    out of the box, session-partitioned so a session's files are
    easy to list together."""
    now = datetime.now(timezone.utc)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    return (
        f"{upload_kind}/{now:%Y/%m/%d}/{session_id or 'no-session'}/"
        f"{uuid.uuid4().hex}_{safe_name}"
    )


def _local_upload_file(
    file_bytes: bytes,
    original_filename: str,
    content_type: str,
    upload_kind: str,
    session_id: Optional[str],
    user_id: Optional[str],
    checksum: str,
    key: str,
) -> dict:
    """Writes file_bytes under LOCAL_STORAGE_DIR/key, mirroring the same
    metadata shape upload_file() returns for S3, so callers and the
    file_uploads table don't need to know which backend was used."""
    local_path = os.path.join(LOCAL_STORAGE_DIR, key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(file_bytes)

    return {
        "session_id": session_id,
        "user_id": user_id,
        "original_filename": original_filename,
        "content_type": content_type,
        "file_size_bytes": len(file_bytes),
        "checksum_sha256": checksum,
        "upload_kind": upload_kind,
        "s3_bucket": "local-disk",
        "s3_key": key,
        "s3_region": "local",
    }


def upload_file(
    file_bytes: bytes,
    original_filename: str,
    content_type: str,
    upload_kind: str,               # 'purchase_order' | 'general_document' | 'audio'
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """Uploads `file_bytes` to S3 and returns metadata ready to hand to
    db.postgres_audit_log.record_file_upload(**metadata, extracted_metadata=...).

    Falls back to local disk storage (with a one-time warning) whenever S3
    isn't configured or a configured bucket can't be reached, instead of
    failing the upload outright.
    """
    checksum = hashlib.sha256(file_bytes).hexdigest()
    key = _build_key(upload_kind, session_id, original_filename)

    if not _s3_configured():
        _warn_local_fallback_once("S3_BUCKET_NAME is not set")
        return _local_upload_file(file_bytes, original_filename, content_type, upload_kind, session_id, user_id, checksum, key)

    tags = {
        "uploaded-by": user_id or "anonymous",
        "session-id": session_id or "none",
    }
    tagging = "&".join(f"{k}={v}" for k, v in tags.items())

    try:
        _client().put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=file_bytes,
            ContentType=content_type,
            ServerSideEncryption="AES256",
            Tagging=tagging,
            Metadata={
                "original-filename": original_filename,
                "uploaded-by": user_id or "anonymous",
                "session-id": session_id or "none",
                "sha256": checksum,
            },
        )
    except (ClientError, NoCredentialsError, EndpointConnectionError) as exc:
        logger.exception("S3 upload failed for key '%s'; falling back to local disk", key)
        _warn_local_fallback_once(f"upload failed: {exc}")
        return _local_upload_file(file_bytes, original_filename, content_type, upload_kind, session_id, user_id, checksum, key)

    return {
        "session_id": session_id,
        "user_id": user_id,
        "original_filename": original_filename,
        "content_type": content_type,
        "file_size_bytes": len(file_bytes),
        "checksum_sha256": checksum,
        "upload_kind": upload_kind,
        "s3_bucket": S3_BUCKET_NAME,
        "s3_key": key,
        "s3_region": AWS_REGION,
    }


def get_presigned_url(s3_key: str, expires_in: int = 3600, s3_bucket: Optional[str] = None) -> str:
    """Read-only presigned URL for pulling a stored file back up (e.g. an
    audit dashboard 'view original document' button). Never make the
    bucket itself public -- presigned URLs are the correct access path.

    For a file stored via the local-disk fallback (s3_bucket == 'local-disk'),
    returns its local file path instead, since there's no S3 object to sign."""
    if s3_bucket == "local-disk" or not _s3_configured():
        return os.path.abspath(os.path.join(LOCAL_STORAGE_DIR, s3_key))
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
        ExpiresIn=expires_in,
    )
