"""
Voice/ws_voice.py — WebSocket voice endpoint

Architecture (Web Speech API mode):
  Browser SpeechRecognition  →  WS {type:"user_speech", text:"..."}
  ↓ stream_agent_turn()  →  WS {type:"token"/"tool_call"/"done"...}
  Browser speechSynthesis reads tokens as they arrive

No audio bytes flow in either direction.
All CRM + Web tools and stream_history.sqlite are shared with the text chat.
"""

import asyncio
import json
import re
import time
from fastapi import WebSocket, WebSocketDisconnect

_ws_bg_tasks = set()

def safe_create_task(coro):
    task = asyncio.create_task(coro)
    _ws_bg_tasks.add(task)
    task.add_done_callback(_ws_bg_tasks.discard)
    return task


def register_voice_ws(app, stream_agent_turn, tts, logger, load_stream_history, save_stream_history, index_touch=None):

    @app.websocket("/ws/voice")
    async def ws_voice(ws: WebSocket, session_id: str = "voice-default", user_id: str = None):
        await ws.accept()
        logger.info("[WS/voice] OPEN  session=%s user=%s", session_id, user_id)

        class ConnectionClosed(Exception):
            """Raised when a background task tries to use a closed websocket."""

        connected = True
        state = {"turn_task": None, "speaking": False}
        send_lock = asyncio.Lock()

        # ------------------------------------------------------------------ #
        # Send helpers                                                         #
        # ------------------------------------------------------------------ #

        async def send_json(payload: dict):
            nonlocal connected
            if not connected:
                raise ConnectionClosed
            try:
                async with send_lock:
                    await ws.send({"type": "websocket.send", "text": json.dumps(payload)})
            except (WebSocketDisconnect, RuntimeError, OSError) as exc:
                connected = False
                raise ConnectionClosed from exc

        # ------------------------------------------------------------------ #
        # Shared conversation history (same sqlite as /api/chat/stream)       #
        # ------------------------------------------------------------------ #
        # (Moved inside run_turn to prevent state duplication across turns)
        # ------------------------------------------------------------------ #
        # Turn management                                                      #
        # ------------------------------------------------------------------ #

        async def cancel_turn():
            task = state["turn_task"]
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("[WS/voice] Turn task raised on cancel")
            state["turn_task"] = None
            state["speaking"] = False

        # ------------------------------------------------------------------ #
        # Text cleaning for TTS (removes markdown, tables, action tags)       #
        # The cleaned text is sent as voice_text events so the browser        #
        # speechSynthesis speaks clean prose instead of raw markdown.         #
        # ------------------------------------------------------------------ #

        _ACTION_TAG_RE = re.compile(r'\[Action:[^\]]*\]')
        _CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
        _TABLE_LINE_RE = re.compile(r'^\s*\|.*\|\s*$')

        def clean_for_speech(text: str) -> str:
            """Strip markdown formatting that sounds awful when spoken."""
            text = _ACTION_TAG_RE.sub('', text)
            text = _CODE_BLOCK_RE.sub('', text)
            lines = [l for l in text.split('\n') if not _TABLE_LINE_RE.match(l)]
            text = '\n'.join(lines)
            text = text.replace('**', '').replace('*', '').replace('#', '')
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # ------------------------------------------------------------------ #
        # Core turn: call agent, stream events back to browser                #
        # ------------------------------------------------------------------ #

        async def run_turn(text: str):
            t0 = time.monotonic()
            logger.info("[WS/voice] turn START  session=%s  text=%r", session_id, text[:120])
            state["speaking"] = True
            token_buf = ""   # accumulates cleaned text for speech
            
            history = await load_stream_history(session_id)
            start_len = len(history)  # snapshot before turn mutates history

            try:
                history = await load_stream_history(session_id)
                async for event in stream_agent_turn(
                    text, session_id=session_id, user_id=user_id, history=history
                ):
                    etype = event["type"]

                    if etype == "token":
                        raw = event["text"]
                        await send_json({"type": "token", "text": raw})
                        token_buf += raw

                        # Emit a voice_sentence whenever a sentence boundary
                        # is detected so speechSynthesis can start speaking
                        # before the full reply is done — low latency.
                        parts = re.split(r'(?<=[.!?\u0964])\s+', token_buf)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                cleaned = clean_for_speech(sentence)
                                if cleaned:
                                    await send_json({"type": "voice_sentence", "text": cleaned})
                            token_buf = parts[-1]

                    elif etype == "tool_call":
                        logger.info(
                            "[WS/voice] tool_call  name=%s  args=%s",
                            event["name"], str(event.get("args", {}))[:200]
                        )
                        await send_json({
                            "type": "tool_call",
                            "name": event["name"],
                            "args": event.get("args", {})
                        })

                    elif etype == "tool_result":
                        logger.info(
                            "[WS/voice] tool_result  name=%s  result=%s",
                            event["name"], str(event.get("result", ""))[:200]
                        )
                        await send_json({
                            "type": "tool_result",
                            "name": event["name"],
                            "result": str(event.get("result", ""))
                        })

                    elif etype == "done":
                        # Flush any remaining text in the buffer
                        final = clean_for_speech(token_buf)
                        if final:
                            await send_json({"type": "voice_sentence", "text": final})
                        token_buf = ""
                        elapsed = (time.monotonic() - t0) * 1000
                        logger.info("[WS/voice] turn DONE  %.0fms  session=%s", elapsed, session_id)
                        await send_json({"type": "done"})

            except asyncio.CancelledError:
                logger.info("[WS/voice] turn CANCELLED  session=%s", session_id)
                raise
            except ConnectionClosed:
                pass
            except Exception as exc:
                logger.exception("[WS/voice] Turn failed: %s", exc)
                try:
                    await send_json({"type": "error", "message": str(exc)})
                except ConnectionClosed:
                    pass
            finally:
                state["speaking"] = False
                # Save ONLY new messages from this turn (delta) — the checkpointer
                # uses add_messages which appends, so saving full history would duplicate.
                delta = history[start_len:]
                if delta:
                    safe_create_task(save_stream_history(session_id, delta))
                    if index_touch:
                        safe_create_task(
                            index_touch(session_id, channel="voice", preview_text=text, increment=len(delta))
                        )

        # ------------------------------------------------------------------ #
        # Main receive loop                                                   #
        # ------------------------------------------------------------------ #

        try:
            while True:
                message = await ws.receive()

                if message.get("type") == "websocket.disconnect":
                    logger.info("[WS/voice] DISCONNECT  session=%s", session_id)
                    connected = False
                    break

                text_msg = message.get("text")
                if not text_msg:
                    continue

                try:
                    control = json.loads(text_msg)
                except Exception:
                    logger.warning("[WS/voice] Bad JSON from client: %r", text_msg[:120])
                    continue

                msg_type = control.get("type")
                logger.debug("[WS/voice] recv  type=%s  session=%s", msg_type, session_id)

                if msg_type == "user_speech":
                    # Web Speech API STT → transcript text arrives here
                    text = (control.get("text") or "").strip()
                    if not text:
                        continue
                    logger.info("[WS/voice] user_speech  %r  session=%s", text[:120], session_id)
                    await cancel_turn()
                    state["turn_task"] = asyncio.create_task(run_turn(text))

                elif msg_type == "interrupt":
                    logger.info("[WS/voice] interrupt requested  session=%s", session_id)
                    await cancel_turn()
                    await send_json({"type": "interrupted"})

                elif msg_type == "end":
                    logger.info("[WS/voice] end received  session=%s", session_id)
                    break

        except (WebSocketDisconnect, ConnectionClosed):
            connected = False
        finally:
            connected = False
            await cancel_turn()
            logger.info("[WS/voice] CLOSED  session=%s", session_id)
