
import logging
import os
import time
import threading
from cachetools import LRUCache
from cachetools.func import lru_cache

from fastapi import HTTPException, APIRouter
import torch
import whisper


LOG_LEVEL = os.getenv("API_LOG_LEVEL", "DEBUG")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("web_stt_demo.api")
logger.setLevel(LOG_LEVEL)

# Shared LRU cache for loaded Whisper models keyed by model name.
# Keep the cache small by default to limit memory usage.
model_cache = LRUCache(maxsize=8)
_model_cache_lock = threading.Lock()

MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")


def get_model(model_name: str = MODEL_NAME):
    """Return a loaded Whisper model for `model_name`.

    If the model is cached, return it immediately. Otherwise load it into
    the shared `model_cache` and return the loaded instance. This function
    may block while loading the model so callers should run it in a worker
    thread if they don't want to block the event loop.
    """
    with _model_cache_lock:
        m = model_cache.get(model_name)
    if m is not None:
        return m

    # Load model (may be slow). We don't attempt a single-flight here; callers
    # who want to avoid duplicate loads should coordinate externally.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Whisper model '%s' on %s...", model_name, device)
    start = time.time()
    m = whisper.load_model(model_name, device=device)
    logger.info("Model %s loaded successfully in %.2fs", model_name, time.time() - start)
    with _model_cache_lock:
        model_cache[model_name] = m
    return m


def is_model_loaded(model_name: str = MODEL_NAME) -> bool:
    """Return True if the given model is currently present in the shared cache."""
    with _model_cache_lock:
        return model_name in model_cache


def load_whisper_model(model_name: str = MODEL_NAME):
    """Backward-compatible wrapper used elsewhere in the codebase."""
    return get_model(model_name)


models_router = APIRouter(prefix="/models", tags=["Models"])

@lru_cache(maxsize=1)
def get_languages_list():
    """Return mapping of Whisper-supported language codes to language names.

    Whisper exposes a LANGUAGES mapping (language name -> code) in
    whisper.tokenizer. We invert that mapping to return code -> name which
    is convenient for clients selecting a language by code.
    """
    try:
        from whisper import tokenizer
        langs = getattr(tokenizer, "LANGUAGES", {}) or {}
        # return the mapping available from whisper; if empty fall back to English
        return langs if langs else {"en": "English"}
    except Exception:
        # If whisper or tokenizer isn't present or fails, default to English
        logger.debug("Whisper tokenizer not available, defaulting languages to English")
        return {"en": "English"}




@models_router.get("/languages")
async def get_languages():
    """Return mapping of Whisper-supported language codes to language names.

    Whisper exposes a LANGUAGES mapping (language name -> code) in
    whisper.tokenizer. We invert that mapping to return code -> name which
    is convenient for clients selecting a language by code.
    """
    return get_languages_list()

@lru_cache(maxsize=1)
def get_models_list():
    """Return a list of Whisper model names that can be used by the client/UI.

    The function will attempt to use library helpers if available, otherwise
    it falls back to a conservative static list of common Whisper models.
    """
    try:
        # Prefer an API from the whisper package if present
        if hasattr(whisper, "available_models") and callable(getattr(whisper, "available_models")):
            try:
                models = whisper.available_models()
                # ensure we return a JSON-serializable list
                return {"models": list(models)}
            except Exception:
                # fall through to other heuristics
                logger.debug("whisper.available_models() exists but failed; falling back")

        # Some whisper versions expose internal _MODELS mapping
        if hasattr(whisper, "_MODELS"):
            try:
                m = getattr(whisper, "_MODELS")
                # _MODELS may be a dict mapping model name -> URL
                if isinstance(m, dict):
                    return {"models": list(m.keys())}
                # otherwise coerce to list
                return {"models": list(m)}
            except Exception:
                logger.debug("whisper._MODELS exists but could not be read")

        # Conservative fallback list of common model sizes
        fallback = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2"]
        return {"models": fallback}

    except Exception as e:
        logger.exception("Failed to enumerate models")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {e}")

@models_router.get("")
async def get_models():
    """Return a list of Whisper model names that can be used by the client/UI.

    The function will attempt to use library helpers if available, otherwise
    it falls back to a conservative static list of common Whisper models.
    """
    return get_models_list()


# Transcribers router: expose available realtime or offline transcribers (whisper, vosk, etc.)
transcribers_router = APIRouter(prefix="/transcribers", tags=["Transcribers"])


# VOSK support removed from API: this module reports Whisper transcribers only.


def get_transcribers_list():
    """Return a list of available transcribers and some metadata for clients.

    The function reports two transcribers by default:
      - whisper: uses the existing Whisper model loader and exposes available model names
      - vosk: reports whether the vosk package and a model directory are present

    This lets a client list choices for session creation.
    """
    # Whisper metadata
    try:
        whisper_models = get_models_list().get("models", [])
    except Exception:
        whisper_models = []
    whisper_entry = {
        "name": "whisper",
        "default_model": MODEL_NAME,
        "available_models": whisper_models,
    }
    return {"transcribers": [whisper_entry]}


@transcribers_router.get("")
async def get_transcribers():
    """Return metadata about available transcribers (whisper, vosk...)."""
    return get_transcribers_list()


@transcribers_router.get("/{name}")
async def get_transcriber(name: str):
    """Return details about a single transcriber by name.

    Returns 404 if the transcriber name is unknown.
    """
    data = get_transcribers_list()
    for t in data.get("transcribers", []):
        if t.get("name") == name:
            return t
    raise HTTPException(status_code=404, detail=f"Unknown transcriber: {name}")


@transcribers_router.get("/{name}/models")
async def get_transcriber_models(name: str):
    """Return available models for a given transcriber.

    - For `whisper` this returns the whisper model list from `get_models_list()`.
    - For `vosk` this returns the discovered model folder name(s) (if present).
    """
    name = (name or "").lower()
    if name == "whisper":
        try:
            return get_models_list()
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to list Whisper models")
    # VOSK removed — only Whisper is supported by this API.
    raise HTTPException(status_code=404, detail=f"Unknown transcriber: {name}")


@transcribers_router.get("/{name}/model/{model}/languages")
async def get_transcriber_model_languages(name: str, model: str):
    """Return language mapping for a specific transcriber model.

    - Whisper: returns the full language mapping from `get_languages_list()` (or English fallback).
    - VOSK: language metadata isn't standardized; default to English.
    """
    name = (name or "").lower()
    if name == "whisper":
        return get_languages_list()
    # VOSK removed — only Whisper is supported by this API.
    raise HTTPException(status_code=404, detail=f"Unknown transcriber: {name}")
