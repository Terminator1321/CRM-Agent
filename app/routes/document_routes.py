"""Reads an uploaded file (any supported type) and stores its extracted text
against session_id for later Q&A or CRM-record creation."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import db.postgres_audit_log as audit_log
from LLM.document_reader import ALL_SUPPORTED_EXTENSIONS, classify_extension, extract_text_generic
from storage import s3_storage

from .. import agent_setup, state
from ..logging_setup import logger

router = APIRouter()


class GeneralDocumentUploadResponse(BaseModel):
    status: str
    filename: str
    message: str
    page_count: int
    extraction_method: str


@router.post("/api/upload-document", response_model=GeneralDocumentUploadResponse)
async def upload_general_document(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    user_id: str = Form("anonymous"),
):
    """Routes by file extension: images/PDFs go through OCR/Vision, plain-text
    formats (.txt/.md/.csv/.docx/.xlsx/...) are read directly."""
    try:
        extension_kind = classify_extension(file.filename)
        if extension_kind == "unsupported":
            supported = ", ".join(sorted(ALL_SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file.filename}'. Supported extensions: {supported}."
            )

        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")

        logger.info(
            f"File '{file.filename}' uploaded for session '{session_id}' "
            f"(routed as '{extension_kind}'). Extracting text..."
        )

        upload_meta = s3_storage.upload_file(
            file_bytes=file_bytes,
            original_filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            upload_kind="general_document",
            session_id=session_id,
            user_id=user_id,
        )

        if extension_kind == "ocr":
            extraction = agent_setup.llm_ocr_engine.extract_document_text(
                file_bytes=file_bytes, mime_type=file.content_type or "application/pdf"
            )
        else:
            extraction = extract_text_generic(file_bytes=file_bytes, filename=file.filename)

        state.document_store[session_id] = {
            "filename": file.filename,
            "text": extraction["text"],
            "injected": False,
        }

        audit_log.record_file_upload(
            **upload_meta,
            extracted_metadata={
                "page_count": extraction["page_count"],
                "pages_read": extraction["pages_read"],
                "method": extraction["method"],
            },
            status="processed",
        )

        return GeneralDocumentUploadResponse(
            status="success",
            filename=file.filename,
            message=(
                f"Document read successfully ({extraction['pages_read']}/{extraction['page_count']} "
                f"page(s), method={extraction['method']}). Ask me a question about it, or tell me to "
                f"create a Lead, Deal, Organization, or Contact from it."
            ),
            page_count=extraction["page_count"],
            extraction_method=extraction["method"],
        )

    except HTTPException as http_exc:
        raise http_exc
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error handling document upload in /api/upload-document")
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")
