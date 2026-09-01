"""The four turn-taking endpoints: /api/tts, /api/tts/stream, /api/chat, /api/chat/stream, /query."""
import base64
import json
import os
import shutil
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import agent_setup, state
from ..graph.nodes import generate_reply, load_stream_history, save_stream_history
from ..logging_setup import logger
from ..streaming import stream_agent_turn

router = APIRouter()


def _get_tts_audio(text: str):
    """Synthesizes text to a WAV file and returns its raw bytes, or None on failure."""
    wav_path = agent_setup.assistant.tts.synthesize_to_file(text)
    try:
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


class TTSRequest(BaseModel):
    text: str


@router.post("/api/tts")
async def synthesize_speech(req: TTSRequest):
    """Synthesizes arbitrary text to speech with the same voice Live Voice Mode uses."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        wav_bytes = _get_tts_audio(text)
    except Exception as exc:
        logger.exception("TTS synthesis failed for /api/tts")
        raise HTTPException(status_code=500, detail=str(exc))

    if not wav_bytes:
        raise HTTPException(status_code=500, detail="TTS synthesis returned no audio")

    return {"audio": base64.b64encode(wav_bytes).decode("ascii")}


@router.post("/api/tts/stream")
async def synthesize_speech_stream(req: TTSRequest):
    """Streams TTS audio chunks directly from OpenAI as they're generated."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    def generate_chunks():
        try:
            for chunk in agent_setup.assistant.tts.synthesize_stream(text, response_format="mp3"):
                yield chunk
        except Exception:
            logger.exception("Streaming TTS failed")

    return StreamingResponse(generate_chunks(), media_type="audio/mpeg")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: Optional[str] = None


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """JSON contract: { message } in, { reply, audio } out."""
    text = (req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        reply = await generate_reply(text, req.session_id, user_id=req.user_id)
    except Exception as exc:
        logger.exception("Agent failed to process message: %s", text)
        raise HTTPException(status_code=500, detail=str(exc))

    audio_b64 = None
    try:
        wav_bytes = _get_tts_audio(reply)
        if wav_bytes:
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    except Exception:
        logger.exception("TTS step failed; returning text-only reply")

    return {"reply": reply, "audio": audio_b64}


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE variant of /api/chat for token-by-token streaming, backed by its
    own per-session_id history in stream_history.sqlite."""
    text = (req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    history = await load_stream_history(req.session_id)
    start_len = len(history)

    async def event_gen():
        # Flush headers through any dev tunnel before the agent pipeline starts,
        # so a slow tool call doesn't look like a CORS failure to the browser.
        yield ": connected\n\n"
        try:
            async for event in stream_agent_turn(text, session_id=req.session_id, user_id=req.user_id, history=history):
                browser_event = {k: v for k, v in event.items() if k != "_delta"}
                yield f"data: {json.dumps(browser_event)}\n\n"
        except Exception as exc:
            logger.exception("Streaming agent turn failed: %s", text)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            delta = history[start_len:]
            if delta:
                state.safe_create_task(save_stream_history(req.session_id, delta))

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query")
async def handle_query(
    query: str = Form(None),
    file: UploadFile = File(None),
    session_id: str = Form("default"),
    user_id: str = Form("anonymous"),
):
    """Text-or-audio variant of /api/chat."""
    if not query and not file:
        raise HTTPException(status_code=400, detail="Either 'query' (text) or 'file' (audio) must be provided.")

    query_text = ""

    if file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Saved uploaded audio file to {temp_file_path}")

            query_text = agent_setup.assistant.whisper.transcribe_file(temp_file_path)
            logger.info(f"Transcribed Audio Text: {query_text}")
        except Exception as e:
            logger.error(f"Error handling uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Error transcribing audio file: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        query_text = query

    logger.info(f"Processing query: '{query_text}'")
    response_text = await generate_reply(query_text, session_id, user_id=user_id)

    audio_b64 = None
    try:
        wav_bytes = _get_tts_audio(response_text)
        if wav_bytes:
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    except Exception:
        logger.exception("TTS step failed; returning text-only reply")

    return {
        "query": query_text,
        "response": response_text,
        "audio": audio_b64,
    }
