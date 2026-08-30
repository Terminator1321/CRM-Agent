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
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("s3-storage")

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

_s3_client = None


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

    Raises botocore.exceptions.ClientError on failure -- callers should
    catch this and turn it into an HTTP 500, same as other upload errors.
    """
    checksum = hashlib.sha256(file_bytes).hexdigest()
    key = _build_key(upload_kind, session_id, original_filename)

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
    except ClientError:
        logger.exception("S3 upload failed for key '%s'", key)
        raise

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


def get_presigned_url(s3_key: str, expires_in: int = 3600) -> str:
    """Read-only presigned URL for pulling a stored file back up (e.g. an
    audit dashboard 'view original document' button). Never make the
    bucket itself public -- presigned URLs are the correct access path."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
        ExpiresIn=expires_in,
    )
