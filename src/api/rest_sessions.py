import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Optional

import pydub
import torch
from cachetools import LRUCache
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from .models import get_model, is_model_loaded, load_whisper_model

LOG_LEVEL = os.getenv("API_LOG_LEVEL", "DEBUG")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("web_stt_demo.api")
logger.setLevel(LOG_LEVEL)

session_router = APIRouter(prefix="/sessions", tags=["Sessions"])

# In-memory session store (LRU) and a simple jobs cache for background transcriptions
sessions_cache = LRUCache(maxsize=512)
_sessions_cache_lock = threading.Lock()

# Jobs cache for asynchronous, non-blocking transcription requests.
# Each job entry is: { status: 'pending'|'done'|'failed', created_at, finished_at?, result?, error? }
jobs_cache = LRUCache(maxsize=1024)
_jobs_cache_lock = threading.Lock()


class SessionCreate(BaseModel):
    model: str
    language: Optional[str] = None
    transcriber: Optional[str] = None


@session_router.post("")
async def create_session(payload: SessionCreate):
    sid = str(uuid.uuid4())
    model_name = payload.model
    language = payload.language
    transcriber = payload.transcriber

    # Store session metadata in the in-memory LRU cache immediately.
    created_at = time.time()
    session_obj = {
        "id": sid,
        "model": model_name,
        "transcriber": transcriber,
        "language": language,
        "created_at": created_at,
        "model_loaded": False,
        # model objects are stored in the shared cache managed by src.api.models
        "transcripts": [],
        "all_segments_text": "",
    }

    # store session in cache
    with _sessions_cache_lock:
        sessions_cache[sid] = session_obj

    # Ensure the requested model is loaded in the shared model cache.
    def _load_model_blocking(name: str):
        # Delegate to models.get_model which loads and caches the model.
        return get_model(name)

    loop = asyncio.get_event_loop()
    try:
        # Use the Whisper model loader for all transcribers. If already loaded,
        # mark session as loaded immediately; otherwise load in a worker thread.
        if is_model_loaded(model_name):
            with _sessions_cache_lock:
                sess = sessions_cache.get(sid)
                if sess is not None:
                    sess["model_loaded"] = True
                    sessions_cache[sid] = sess
        else:
            try:
                await loop.run_in_executor(None, _load_model_blocking, model_name)
                with _sessions_cache_lock:
                    sess = sessions_cache.get(sid)
                    if sess is not None:
                        sess["model_loaded"] = True
                        sessions_cache[sid] = sess
            except Exception:
                logger.exception(
                    "Failed to load model %s for session %s", model_name, sid
                )
    except Exception:
        # guard outer try in case event loop lookup fails
        logger.exception("Failed to start model load for %s", model_name)

    return {
        "session_id": sid,
        "model": model_name,
        "transcriber": transcriber,
        "language": language,
        "created_at": created_at,
    }


@session_router.get("/{session_id}")
async def get_session_status(session_id: str):
    """Return session metadata for the session.

    Response shape:
    {
      "id": <session_id>,
      "model": <model_name>,
      "language": <language>,
      "created_at": <ts>,
      "model_loaded": <bool>
    }
    """
    try:
        with _sessions_cache_lock:
            session = sessions_cache.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        model_loaded = session.get("model_loaded", False)

        return {
            "id": session["id"],
            "model": session["model"],
            "language": session["language"],
            "created_at": session["created_at"],
            "model_loaded": bool(model_loaded),
            "transcripts": session.get("transcripts", []),
            "all_segments_text": session.get("all_segments_text", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch session %s", session_id)
        raise HTTPException(status_code=500, detail="Failed to fetch session") from exc


def _blocking_transcribe_and_persist(
    content: bytes,
    filename: Optional[str],
    language: Optional[str],
    session_id: Optional[str],
):
    """Blocking helper to write uploaded bytes to temp files, normalize audio,
    and run whisper transcription.

    This runs on a worker thread via run_in_executor to avoid blocking the event loop.
    Returns a dict with keys: text, language, segments
    """
    if not content:
        raise RuntimeError("No file content to transcribe")

    suffix = os.path.splitext(filename or "upload.wav")[1] or ".wav"
    tmp_file_path = None
    normalized_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        logger.info(
            "Received upload '%s' (%d bytes); saved to %s",
            filename,
            len(content),
            tmp_file_path,
        )

        # Choose Whisper model for this request. Use the shared model cache helper.
        whisper_model = None
        model_name_for_request = None
        sess = None
        if session_id:
            with _sessions_cache_lock:
                sess = sessions_cache.get(session_id)

        model_name_for_request = sess.get("model") if sess is not None else None

        if model_name_for_request:
            try:
                whisper_model = get_model(model_name_for_request)
            except Exception:
                logger.exception(
                    "Failed to load model %s for transcription",
                    model_name_for_request,
                )

        # Load API default if none was selected/loaded
        if whisper_model is None:
            whisper_model = load_whisper_model()

        logger.info(
            "Starting transcription for %s (language=%s, session=%s)",
            tmp_file_path,
            language,
            session_id,
        )

        # Normalize audio to mono 16 kHz WAV before passing to Whisper.
        try:
            audio = pydub.AudioSegment.from_file(tmp_file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as norm_tmp:
                audio.export(norm_tmp.name, format="wav")
                normalized_path = norm_tmp.name
            logger.debug("Normalized audio saved to %s", normalized_path)
        except Exception as e:
            logger.warning(
                "Audio normalization failed, proceeding with original file: %s", e
            )

        input_path = normalized_path or tmp_file_path

        # (VOSK removed) use Whisper snapshot transcription

        # fallback: use whisper snapshot transcription
        t0 = time.time()
        if whisper_model:
            result = whisper_model.transcribe(input_path, language=language, fp16=torch.cuda.is_available())        
            elapsed = time.time() - t0
            logger.info("Transcription finished in %.2fs", elapsed)

            text = result.get("text", "")
            segments = result.get("segments", [])

        # Transactional: do not persist or mutate session state here.
        return {"text": text, "language": result.get("language"), "segments": segments}
    except Exception as e:
        logger.exception("Transcription helper failed: %s", e)
        raise RuntimeError(str(e))


def _background_run_transcription(
    job_id: str,
    content: bytes,
    filename: Optional[str],
    language: Optional[str],
    session_id: Optional[str],
):
    """Run blocking transcription in a worker thread and store result in jobs_cache.

    This function is intended to be executed in an executor (threadpool) so it may
    call the blocking transcription helper directly.
    """
    try:
        result = _blocking_transcribe_and_persist(
            content, filename, language, session_id
        )
        with _jobs_cache_lock:
            jobs_cache[job_id] = {
                "status": "done",
                "created_at": jobs_cache.get(job_id, {}).get("created_at", time.time()),
                "finished_at": time.time(),
                "result": result,
            }
    except Exception as e:
        logger.exception("Background transcription job %s failed: %s", job_id, e)
        with _jobs_cache_lock:
            jobs_cache[job_id] = {
                "status": "failed",
                "created_at": jobs_cache.get(job_id, {}).get("created_at", time.time()),
                "finished_at": time.time(),
                "error": str(e),
            }


@session_router.post("/{session_id}/transcribe")
async def transcribe_for_session(session_id: str, file: UploadFile = File(...)):
    """Session-scoped transcription endpoint. Requires a valid session_id.

    The language for transcription is read from the session metadata.
    Runs blocking work in a threadpool to avoid blocking the event loop.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Lookup session in the in-memory cache and read language
    try:
        with _sessions_cache_lock:
            session = sessions_cache.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session_language = session.get("language")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to read session %s before transcribe", session_id)
        raise HTTPException(status_code=500, detail="Failed to read session") from exc

    # Read content from the upload. Scheduling the heavy work to run in the
    # background so this request can return quickly for real-time clients.
    content = await file.read()

    # Create an asynchronous job id and store a pending entry in the jobs cache.
    job_id = str(uuid.uuid4())
    with _jobs_cache_lock:
        jobs_cache[job_id] = {
            "status": "pending",
            "created_at": time.time(),
            "session_id": session_id,
        }

    loop = asyncio.get_event_loop()
    # Schedule the blocking work in the default threadpool; don't await it here.
    try:
        loop.run_in_executor(
            None,
            _background_run_transcription,
            job_id,
            content,
            file.filename,
            session_language,
            session_id,
        )
        return {"job_id": job_id, "status": "accepted"}
    except Exception:
        logger.exception(
            "Failed to schedule transcription job for session %s", session_id
        )
        # mark job as failed if scheduling itself fails
        with _jobs_cache_lock:
            jobs_cache[job_id] = {
                "status": "failed",
                "created_at": time.time(),
                "finished_at": time.time(),
                "error": "scheduling_failed",
            }
        raise HTTPException(status_code=500, detail="Failed to schedule transcription")


@session_router.get("/transcribe/jobs/{job_id}")
async def get_transcription_job(job_id: str):
    """Retrieve the status/result of an asynchronous transcription job."""
    try:
        with _jobs_cache_lock:
            job = jobs_cache.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch job %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to fetch job") from exc


# @sessions.post("/transcribe")
# async def transcribe_audio(
#     file: UploadFile = File(...),
#     language: Optional[str] = None
# ):
#     """
#     Transcribe audio file using Whisper.

#     Args:
#         file: Audio file (wav, mp3, m4a, etc.)
#         language: Optional language code (e.g., 'en', 'es')

#     Returns:
#         Transcription result with text and metadata
#     """
#     if not file:
#         raise HTTPException(status_code=400, detail="No file provided")

#     # Save uploaded file temporarily
#     content = await file.read()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
#         tmp_file.write(content)
#         tmp_file_path = tmp_file.name
#     logger.info("Received upload '%s' (%d bytes); saved to %s", file.filename, len(content), tmp_file_path)

#     try:
#         # Ensure model is loaded
#         whisper_model = load_whisper_model()

#         # Transcribe
#         logger.info("Starting transcription for %s (language=%s)", tmp_file_path, language)

#         # Normalize audio to mono 16 kHz WAV before passing to Whisper.
#         # This avoids ffmpeg/decoder errors when channel layouts are unexpected.
#         normalized_path = None
#         try:
#             audio = pydub.AudioSegment.from_file(tmp_file_path)
#             audio = audio.set_frame_rate(16000).set_channels(1)
#             norm_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#             audio.export(norm_tmp.name, format="wav")
#             normalized_path = norm_tmp.name
#             logger.debug("Normalized audio saved to %s", normalized_path)
#         except Exception as e:
#             # If normalization fails, fall back to original file and log warning
#             logger.warning("Audio normalization failed, proceeding with original file: %s", e)

#         input_path = normalized_path or tmp_file_path
#         t0 = time.time()
#         result = whisper_model.transcribe(
#             input_path,
#             language=language,
#             fp16=torch.cuda.is_available()
#         )
#         elapsed = time.time() - t0
#         logger.info("Transcription finished in %.2fs", elapsed)

#         return {
#             "text": result["text"],
#             "language": result.get("language"),
#             "segments": result.get("segments", [])
#         }

#     except Exception as e:
#         logger.exception("Transcription failed for %s", tmp_file_path)
#         raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

#     finally:
#         # Clean up temporary file
#         if os.path.exists(tmp_file_path):
#             os.unlink(tmp_file_path)
#         if 'normalized_path' in locals() and normalized_path and os.path.exists(normalized_path):
#             try:
#                 os.unlink(normalized_path)
#             except Exception:
#                 pass
