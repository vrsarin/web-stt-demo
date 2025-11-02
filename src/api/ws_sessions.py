import json
import asyncio
import logging
import os
import tempfile
import time
from typing import Optional

import pydub
import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .models import get_model, is_model_loaded, load_whisper_model
from .rest_sessions import sessions_cache, _sessions_cache_lock
LOG_LEVEL = os.getenv("API_LOG_LEVEL", "DEBUG")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("web_stt_demo.api.ws")
logger.setLevel(LOG_LEVEL)

ws_router = APIRouter(prefix="/ws", tags=["WS Sessions"]) 


def _blocking_transcribe_bytes(
    content: bytes,
    filename: Optional[str],
    language: Optional[str],
    session_id: Optional[str],
):
    """Blocking helper that mirrors rest_sessions._blocking_transcribe_and_persist

    Accepts raw uploaded bytes (as the REST handler did) and returns a dict
    containing text, language and segments.
    """
    if not content:
        raise RuntimeError("No file content to transcribe")

    suffix = os.path.splitext(filename or "upload.wav")[1] or ".wav"
    tmp_file_path = None
    normalized_path = None
    try:
        # write bytes to a temp file so pydub/whisper can open it
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        logger.debug("WS: Saved incoming audio to %s (session=%s)", tmp_file_path, session_id)

        # determine model for session if available
        whisper_model = None
        model_name_for_request = None
        if session_id:
            with _sessions_cache_lock:
                sess = sessions_cache.get(session_id)
            model_name_for_request = sess.get("model") if sess is not None else None

        if model_name_for_request:
            try:
                whisper_model = get_model(model_name_for_request)
            except Exception:
                logger.exception("Failed to load model %s for transcription", model_name_for_request)

        if whisper_model is None:
            whisper_model = load_whisper_model()

        # Normalize audio to mono 16k WAV where possible
        try:
            audio = pydub.AudioSegment.from_file(tmp_file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as norm_tmp:
                audio.export(norm_tmp.name, format="wav")
                normalized_path = norm_tmp.name
            logger.debug("WS: Normalized audio saved to %s", normalized_path)
        except Exception as e:
            logger.debug("WS: Audio normalization failed, will try original file: %s", e)

        input_path = normalized_path or tmp_file_path

        # Quick guard: skip transcription for very short / empty audio. Whisper
        # can error when given zero-length inputs (reshape of empty tensors).
        try:
            _probe_audio = pydub.AudioSegment.from_file(input_path)
            duration_ms = len(_probe_audio)
            if duration_ms <= 50:
                logger.debug("WS: audio too short (%.1f ms), skipping transcription (session=%s)", duration_ms, session_id)
                return {"text": "", "language": None, "segments": []}
        except Exception:
            # If probing fails, continue and let the downstream logic handle it.
            pass

        # (VOSK removed) proceed with Whisper transcription path

        # fallback: use whisper snapshot transcription
        if whisper_model is None:
            logger.error("WS: no Whisper model available for session %s; skipping transcription", session_id)
            return {"text": "", "language": None, "segments": []}

        t0 = time.time()

        # Defensive audio cleaning: read the normalized file and guard against
        # NaNs or invalid sample values which can cause Whisper to produce NaN
        # logits (seen in some corrupted or unusual inputs). If we detect
        # non-finite samples, replace them with zeros and write a cleaned
        # temporary WAV to pass to the model.
        try:
            audio_check = pydub.AudioSegment.from_file(input_path)
            # sample width -> dtype
            sw = audio_check.sample_width
            if sw == 2:
                dtype = np.int16
            elif sw == 4:
                dtype = np.int32
            else:
                dtype = np.int16

            samples = np.frombuffer(audio_check.raw_data, dtype=dtype)
            # promote to float for isnan/isfinite checks
            samples_f = samples.astype(np.float32)
            if not np.isfinite(samples_f).all():
                logger.warning("WS: Detected non-finite samples in audio; cleaning and retrying (session=%s)", session_id)
                samples_f[~np.isfinite(samples_f)] = 0.0
                # convert back to original integer dtype
                cleaned = samples_f.astype(dtype)
                cleaned_bytes = cleaned.tobytes()
                cleaned_seg = audio_check._spawn(cleaned_bytes)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ct:
                    cleaned_seg.export(ct.name, format="wav")
                    input_path = ct.name
        except Exception:
            # If anything goes wrong during cleaning, log and proceed with
            # the original input_path — the downstream try/except will catch
            # errors from Whisper.
            logger.debug("WS: audio cleaning step failed or not needed (session=%s)", session_id)

        result = whisper_model.transcribe(input_path, language=language, fp16=torch.cuda.is_available())
        elapsed = time.time() - t0
        logger.debug("WS: Transcription finished in %.2fs (session=%s)", elapsed, session_id)

        text = result.get("text", "")
        segments = result.get("segments", [])

        return {"text": text, "language": result.get("language"), "segments": segments}
    except Exception as e:
        logger.exception("WS: Transcription helper failed: %s", e)
        raise RuntimeError(str(e))
    finally:
        # Clean up temporary files where possible
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except Exception:
            pass
        try:
            if normalized_path and os.path.exists(normalized_path):
                os.unlink(normalized_path)
        except Exception:
            pass


# --- Transcriber adapter (placeholder for streaming-friendly backends) ---
class TranscriberAdapter:
    """Simple adapter interface so we can later swap in streaming-friendly ASR.

    Current implementation delegates to the blocking whisper helper.
    """

    def __init__(self, session_id: Optional[str], language: Optional[str]):
        self.session_id = session_id
        self.language = language

    def transcribe_bytes_blocking(self, content: bytes):
        """Blocking transcription wrapper used by executor."""
        return _blocking_transcribe_bytes(content, "upload.wav", self.language, self.session_id)


async def transcribe_snapshot_and_send(websocket: WebSocket, snapshot: bytes, session_id: str, session_language: Optional[str], is_final: bool = False):
    """Run blocking transcription in executor, persist and send results over websocket."""
    loop = asyncio.get_event_loop()
    adapter = TranscriberAdapter(session_id, session_language)

    try:
        result = await loop.run_in_executor(None, adapter.transcribe_bytes_blocking, snapshot)
    except Exception as e:
        logger.exception("WS: transcription failed for session %s: %s", session_id, e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        return

    # Persist into session transcripts (thread-safe)
    try:
        with _sessions_cache_lock:
            sess = sessions_cache.get(session_id)
            if sess is not None:
                transcripts = sess.get("transcripts", [])
                transcripts.append({"text": result.get("text", ""), "ts": time.time()})
                sess["transcripts"] = transcripts
                sess["all_segments_text"] = sess.get("all_segments_text", "") + "\n" + result.get("text", "")
                sessions_cache[session_id] = sess
    except Exception:
        logger.exception("WS: Failed to persist transcript for session %s", session_id)

    # send back partial or final
    msg_type = "final" if is_final else "partial"
    try:
        await websocket.send_json({"type": msg_type, "text": result.get("text", ""), "segments": result.get("segments", [])})
    except Exception:
        # ignore send errors (client may have disconnected)
        pass


@ws_router.websocket("/sessions/{session_id}")
async def websocket_session(session_id: str, websocket: WebSocket):
    """WebSocket endpoint for realtime/bidirectional transcription.

    Protocol (simple):
      - Client connects to /ws/sessions/{session_id}
      - Client may send binary frames containing audio bytes; those are appended
        to an internal buffer.
      - Client may send JSON control messages (text frames) with shape:
          {"type":"control", "action":"end"}  -> triggers final transcription
          {"type":"control", "action":"ping"}  -> server replies with pong
      - Server runs transcription in a worker thread periodically (when buffer
        grows or when client sends 'end') and sends back JSON messages:
          {"type":"partial", "text":..., "segments": [...]}
          {"type":"final", "text":..., "segments": [...]}  (on end)

    This is a pragmatic implementation: Whisper doesn't provide frame-level
    streaming natively, so we snapshot the accumulated bytes and transcribe
    them using the existing blocking helper in an executor.
    """
    await websocket.accept()

    # Validate session
    with _sessions_cache_lock:
        session = sessions_cache.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "session not found"})
        await websocket.close(code=1000)
        return

    session_language = session.get("language")
    # Ensure model is loaded (non-blocking from websocket perspective)
    loop = asyncio.get_event_loop()
    model_name = session.get("model")
    if model_name and not is_model_loaded(model_name):
        # kick off model load in executor but don't await here; we will load on-demand
        loop.run_in_executor(None, get_model, model_name)

    buffer = bytearray()
    last_snapshot_time = time.time()
    snapshot_interval = float(os.getenv("WS_SNAPSHOT_INTERVAL", "2.0"))
    min_bytes_for_snapshot = int(os.getenv("WS_MIN_BYTES", "16000"))

    # Ensure we don't run concurrent transcription jobs per connection
    transcribe_lock = asyncio.Lock()

    last_partial_text = ""

    async def handle_binary_frame(chunk: bytes, last_snapshot_time: float):
        """Extend buffer and schedule a background transcription task when appropriate.

        Returns updated last_snapshot_time.
        """
        nonlocal last_partial_text
        if not chunk:
            return last_snapshot_time

        # Whisper snapshot: append to buffer and schedule snapshot transcription
        if chunk:
            buffer.extend(chunk)

        now = time.time()
        if len(buffer) >= min_bytes_for_snapshot or (now - last_snapshot_time) >= snapshot_interval:
            if not transcribe_lock.locked():
                async def _task():
                    async with transcribe_lock:
                        # snapshot current buffer
                        snapshot = bytes(buffer)
                        await transcribe_snapshot_and_send(websocket, snapshot, session_id, session_language, is_final=False)

                # schedule and keep a short-lived reference
                _bg = asyncio.create_task(_task())
                last_snapshot_time = now
        return last_snapshot_time

    async def handle_text_frame(text: str):
        """Handle control messages sent as text frames."""
        if not isinstance(text, str):
            return None
        try:
            payload = json.loads(text)
        except Exception:
            if text.strip().lower() == "ping":
                await websocket.send_text("pong")
                return None
            await websocket.send_json({"type": "error", "message": "invalid json"})
            return None

        t = payload.get("type")
        if t != "control":
            await websocket.send_json({"type": "error", "message": "unsupported message type"})
            return None

        action = payload.get("action")
        if action == "end":
            # run final transcription on whole buffer and close
            async with transcribe_lock:
                # Snapshot buffer and run transcription helper.
                snapshot = bytes(buffer)
                if not snapshot:
                    # Nothing to transcribe; just close gracefully.
                    try:
                        await websocket.send_json({"type": "done"})
                    except Exception:
                        pass
                    try:
                        await websocket.close()
                    except Exception:
                        pass
                    return "close"

                try:
                    await transcribe_snapshot_and_send(websocket, snapshot, session_id, session_language, is_final=True)
                except Exception:
                    logger.exception("WS: transcription failed for session %s", session_id)

            # Try to notify client and close - these may fail if the client already closed
            try:
                await websocket.send_json({"type": "done"})
            except Exception:
                pass
            try:
                await websocket.close()
            except Exception:
                pass
            return "close"
        if action == "ping":
            await websocket.send_json({"type": "pong"})
            return None
        if action == "reset":
            buffer.clear()
            await websocket.send_json({"type": "reset_ok"})
            return None
        await websocket.send_json({"type": "error", "message": "unknown action"})
        return None

    try:
        while True:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                logger.debug("WS: client disconnected (session=%s)", session_id)
                break
            except Exception as e:
                # Some transports raise a RuntimeError when a disconnect has
                # already been received; treat that specific case as a normal
                # client disconnect and stop the loop cleanly without a full
                # exception log to avoid noisy duplicate messages.
                msg = str(e or "")
                if isinstance(e, RuntimeError) and "Cannot call \"receive\" once a disconnect message has been received" in msg:
                    logger.debug("WS: receive after disconnect; stopping loop (session=%s)", session_id)
                    break
                # For other errors, log at debug level and stop the loop.
                logger.debug("WS: receive failed (session=%s): %s", session_id, e)
                break

            # Binary audio frames
            if data.get("bytes") is not None:
                chunk = data.get("bytes")
                if not isinstance(chunk, (bytes, bytearray)):
                    # ignore non-bytes frames
                    continue
                last_snapshot_time = await handle_binary_frame(chunk, last_snapshot_time)

            # Text frames: JSON control messages
            elif data.get("text") is not None:
                text = data.get("text")
                if not isinstance(text, str):
                    # ignore non-string text frames
                    continue
                res = await handle_text_frame(text)
                if res == "close":
                    break

            else:
                # unknown frame type; ignore
                continue

    except Exception as e:
        logger.exception("WS: unexpected error in websocket handler for %s: %s", session_id, e)
        try:
            await websocket.close()
        except Exception:
            pass
