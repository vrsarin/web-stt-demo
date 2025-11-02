import io
import json
import os
import threading
import time
import traceback
from typing import Any, Dict, List, Optional
from queue import Queue, Full
import asyncio
import httpx
import websockets


import pydub
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)
from streamlit_webrtc.config import RTCIceServer

from src.ui.audio_processor import AudioProcessor

API_URL = os.getenv("API_URL", "http://localhost:8080")
SAMPLE_RATE = 16000


@st.cache_data
def fetch_api_info(api_url: str):
    """Cached fetch for API /info endpoint."""
    try:
        resp = requests.get(f"{api_url}/info", timeout=5)
        if resp.status_code == 200:
            return resp.json() or {}
        return None
    except Exception:
        return None

@st.cache_data
def fetch_model_languages(api_url: str, transcriber: Optional[str] = None, model: Optional[str] = None):
    """Cached fetch for model languages.

    Prefer a transcriber+model specific endpoint if available. Try common
    endpoint patterns and fall back to the global /models/languages.

    Assumptions:
    - The backend may expose one of these endpoints:
        /transcribers/{transcriber}/models/{model}/languages
        /transcribers/{transcriber}/model/{model}/languages
      If neither exists, we try /models/languages as a final fallback.
    """
    # Try transcriber+model specific endpoints first when provided
    try:
        if transcriber and model:
            # normalize simple names (do not URL-encode aggressively here)
            t = str(transcriber)
            m = str(model)
            candidates = [
                f"{api_url}/transcribers/{t}/models/{m}/languages",
                f"{api_url}/transcribers/{t}/model/{m}/languages",
            ]
            for url in candidates:
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        resp.raise_for_status()
                        return resp.json() or {}
                except Exception:
                    # try next candidate
                    pass

        
        # transcriber+model endpoints did not return data, return empty dict.
        return {}
    except Exception:
        return {}


@st.cache_data
def fetch_transcribers(api_url: str):
    """Cached fetch for API /transcribers endpoint.

    Returns a list of transcribers. Accepts either a JSON list or a dict
    containing a 'transcribers' key.
    """
    try:
        resp = requests.get(f"{api_url}/transcribers", timeout=5)
        resp.raise_for_status()
        jr = resp.json() or {}
        if isinstance(jr, dict) and "transcribers" in jr:
            return jr.get("transcribers") or []
        if isinstance(jr, list):
            return jr
        return []
    except Exception:
        return []


def _submit_transcription_job(audio_data: bytes, language: Optional[str], session_id: Optional[str]):
    """Submit a transcription job to the API.

    If session_id is provided, POST to the session-scoped endpoint which
    returns an async job_id. If no session_id, fall back to the synchronous
    /transcribe endpoint and return the result dict directly.
    Returns job_id (str) or result dict or None on failure.
    """
    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    params = {}
    if language:
        params["language"] = language
    # include selected transcriber when provided via session or queue
    try:
        transcriber = None
        # allow callers to pass transcriber via kwargs in future; fallback to session state
        transcriber = globals().get("_queued_transcriber")
    except Exception:
        transcriber = None
    if not transcriber:
        try:
            transcriber = st.session_state.get("selected_transcriber")
        except Exception:
            transcriber = None
    if transcriber:
        params["transcriber"] = transcriber

    try:
        if session_id:
            try:
                resp = requests.post(
                    f"{API_URL}/sessions/{session_id}/transcribe",
                    files=files,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                jr = resp.json() or {}
                return jr.get("job_id")
            except requests.exceptions.RequestException:
                return None
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/transcribe", files=files, params=params, timeout=60
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException:
                return None
    except Exception:
        return None


def _submit_realtime_transcription(
    audio_data: bytes,
    language: Optional[str],
    session_id: Optional[str],
    transcriber: Optional[str],
    model: Optional[str] = None,
):
    """Submit a realtime transcription using session-scoped endpoints.

    If session_id is None, create a transient session (POST /sessions)
    with the selected model/language/transcriber and use it. Then POST
    the WAV to /sessions/{session_id}/transcribe which returns a job_id.
    Poll the job status endpoint until done/failed and return the result
    dict or None on failure.
    """
    files = {"file": ("audio.wav", audio_data, "audio/wav")}

    # Ensure we have a session to POST to
    sid = session_id
    if not sid:
        # Create a short-lived session for realtime if none exists.
        payload = {"model": model or st.session_state.get("selected_model"), "language": language, "transcriber": transcriber}
        try:
            resp = requests.post(f"{API_URL}/sessions", json=payload, timeout=10)
            resp.raise_for_status()
            jr = resp.json() or {}
            sid = jr.get("session_id")
        except Exception:
            return None

    # POST the file to the session transcribe endpoint which returns a job_id
    try:
        resp = requests.post(f"{API_URL}/sessions/{sid}/transcribe", files=files, timeout=10)
        resp.raise_for_status()
        jr = resp.json() or {}
        job_id = jr.get("job_id")
        if not job_id:
            return None
    except Exception:
        return None

    # Poll for job completion
    poll_url = f"{API_URL}/sessions/transcribe/jobs/{job_id}"
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            pj = requests.get(poll_url, timeout=10)
            pj.raise_for_status()
            job = pj.json() or {}
            status = job.get("status")
            if status == "done":
                return job.get("result")
            if status == "failed":
                return None
        except Exception:
            pass
        time.sleep(0.5)
    return None

# Module-global snapshot store used by the debug HTTP server
_debug_snapshot: Dict[str, Any] = {}

# Module-global references to avoid calling Streamlit APIs from background threads
_webrtc_ctx_ref = None
_last_audio_proc_ref = None
_debug_enabled = False

# Realtime worker queue & results (module-global so background thread
# doesn't write Streamlit session_state directly).
_realtime_queue: "Queue[tuple]" = Queue(maxsize=8)
_realtime_results: List[Optional[dict]] = []
_realtime_results_lock = threading.Lock()
_realtime_worker_started = False

# Async loop and client for non-blocking realtime HTTP I/O
_realtime_async_loop: Optional[asyncio.AbstractEventLoop] = None
_realtime_async_loop_thread: Optional[threading.Thread] = None
_realtime_async_client: Optional[httpx.AsyncClient] = None
_realtime_async_semaphore: Optional[asyncio.Semaphore] = None


def _start_realtime_async_loop(concurrency: int = 3):
    """Start an asyncio event loop in a dedicated thread with an httpx.AsyncClient.

    This loop is used to schedule non-blocking HTTP uploads and polling so the
    realtime submitter can keep consuming queued items without blocking on I/O.
    """
    global _realtime_async_loop, _realtime_async_loop_thread, _realtime_async_client, _realtime_async_semaphore
    if _realtime_async_loop is not None:
        return

    def _run_loop(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    _realtime_async_loop = loop
    # Modern asyncio.Semaphore does not accept 'loop' — create it normally.
    # Semaphores are not bound to an event loop in recent Python versions.
    _realtime_async_semaphore = asyncio.Semaphore(concurrency)

    # Start the loop in a thread
    t = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    t.start()
    _realtime_async_loop_thread = t

    # Create an AsyncClient tied to this loop
    async def _create_client():
        nonlocal loop
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        return client

    # Use run_coroutine_threadsafe to create the client in the loop
    fut = asyncio.run_coroutine_threadsafe(_create_client(), loop)
    try:
        _realtime_async_client = fut.result(timeout=5.0)
    except Exception:
        _realtime_async_client = None


async def _process_realtime_item_async(wav_bytes: bytes, language: Optional[str], session_id: Optional[str], transcriber: Optional[str], model: Optional[str] = None, transport: str = "REST"):
    """Async task to upload a WAV and poll job status, appending result to _realtime_results."""
    global _realtime_async_client, _realtime_async_semaphore
    client = _realtime_async_client
    if client is None:
        return None

    sem = _realtime_async_semaphore
    if sem is None:
        sem = asyncio.Semaphore(3)

    async with sem:
        # For REST transport: ensure session exists and upload via HTTP
        sid = session_id
        if transport == "REST":
            if not sid:
                # Avoid accessing Streamlit session_state from the async background
                # loop; the caller should pass the model if available. Use the
                # provided model value directly.
                payload = {"model": model, "language": language, "transcriber": transcriber}
                try:
                    r = await client.post(f"{API_URL}/sessions", json=payload)
                    r.raise_for_status()
                    jr = r.json() or {}
                    sid = jr.get("session_id")
                except Exception:
                    return None

            # Upload file to session transcribe and record debug info
            try:
                files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
                r2 = await client.post(f"{API_URL}/sessions/{sid}/transcribe", files=files)

                # Capture response status/body for debugging before raising for status
                try:
                    ds = globals().get("_debug_snapshot")
                    if isinstance(ds, dict):
                        # store as simple serializables
                        ds["last_upload_status"] = getattr(r2, "status_code", None)
                        try:
                            ds["last_upload_body"] = r2.text
                        except Exception:
                            # fallback to raw bytes if text decoding fails
                            try:
                                ds["last_upload_body"] = (await r2.aread()).decode("utf-8", errors="replace")
                            except Exception:
                                ds["last_upload_body"] = None
                except Exception:
                    pass

                r2.raise_for_status()
                jr2 = r2.json() or {}
                job_id = jr2.get("job_id")
                if not job_id:
                    return None
            except Exception as e:
                # Record exception for debugging
                try:
                    ds = globals().get("_debug_snapshot")
                    if isinstance(ds, dict):
                        ds["last_upload_exception"] = f"{type(e).__name__}: {str(e)}"
                except Exception:
                    pass
                return None

            # Poll job
            poll_url = f"{API_URL}/sessions/transcribe/jobs/{job_id}"
            deadline = time.time() + 60.0
            while time.time() < deadline:
                try:
                    pj = await client.get(poll_url)
                    pj.raise_for_status()
                    job = pj.json() or {}
                    status = job.get("status")
                    try:
                        ds = globals().get("_debug_snapshot")
                        if isinstance(ds, dict):
                            ds["last_polled_job"] = job_id
                            ds["last_polled_status"] = status
                    except Exception:
                        pass
                    if status == "done":
                        result = job.get("result")
                        try:
                            with _realtime_results_lock:
                                _realtime_results.append(result)
                        except Exception:
                            pass
                        return result
                    if status == "failed":
                        try:
                            ds = globals().get("_debug_snapshot")
                            if isinstance(ds, dict):
                                failures = ds.get('recent_job_failures', [])
                                failures.append({'job_id': job_id, 'ts': time.time(), 'job': job})
                                ds['recent_job_failures'] = failures[-5:]
                        except Exception:
                            pass
                        return None
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            return None
        elif transport == "WebSocket":
            # Use a websocket connection to send bytes and collect partials/final
            # Ensure session exists
            sid = session_id
            if not sid:
                payload = {"model": model, "language": language, "transcriber": transcriber}
                try:
                    r = await client.post(f"{API_URL}/sessions", json=payload)
                    r.raise_for_status()
                    jr = r.json() or {}
                    sid = jr.get("session_id")
                except Exception:
                    return None

            # Build ws URL
            ws_url = None
            try:
                if API_URL.startswith("https://"):
                    ws_url = API_URL.replace("https://", "wss://")
                elif API_URL.startswith("http://"):
                    ws_url = API_URL.replace("http://", "ws://")
                else:
                    ws_url = API_URL
                # ensure no trailing slash
                ws_url = ws_url.rstrip("/") + f"/ws/sessions/{sid}"
            except Exception:
                return None

            try:
                async with websockets.connect(ws_url) as ws:
                    # send binary audio
                    await ws.send(wav_bytes)
                    # read responses for a short window and append any partial/final results
                    try:
                        while True:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                            except asyncio.TimeoutError:
                                break
                            if not msg:
                                continue
                            # messages expected as JSON text frames
                            if isinstance(msg, (bytes, bytearray)):
                                # ignore unexpected binary frames
                                continue
                            try:
                                obj = json.loads(msg)
                                mtype = obj.get("type")
                                if mtype in ("partial", "final"):
                                    with _realtime_results_lock:
                                        _realtime_results.append(obj)
                            except Exception:
                                # non-json text or unexpected; ignore
                                pass
                    except Exception:
                        pass
            except Exception:
                return None
            return None


def _start_realtime_worker():
    """Start the realtime background worker thread once."""
    global _realtime_worker_started
    if _realtime_worker_started:
        return

    def _submitter_worker():
        """Consume WAVs from _realtime_queue and submit transcription jobs.

        For backends that return an async job_id when using sessions, this
        submitter now polls the job status inline (no separate poller thread
        or job-poll queue). Immediate results (dicts) are appended to
        _realtime_results as before.
        """
        while True:
            try:
                # update debug snapshot with current queue sizes
                try:
                    ds = globals().get("_debug_snapshot")
                    if isinstance(ds, dict):
                        ds["realtime_queue_size"] = _realtime_queue.qsize()
                        ds["realtime_worker_started"] = True
                except Exception:
                    pass

                item = _realtime_queue.get()
                if item is None:
                    time.sleep(0.01)
                    continue

                # item expected as (wav_bytes, language, session_id, transcriber, model, transport)
                wav_bytes = None
                language = None
                session_id = None
                transcriber = None
                model = None
                transport = "REST"
                try:
                    # unpack flexibly to support older tuples with fewer elements
                    if isinstance(item, (list, tuple)):
                        if len(item) >= 1:
                            wav_bytes = item[0]
                        if len(item) >= 2:
                            language = item[1]
                        if len(item) >= 3:
                            session_id = item[2]
                        if len(item) >= 4:
                            transcriber = item[3]
                        if len(item) >= 5:
                            model = item[4]
                        if len(item) >= 6:
                            transport = item[5]
                    else:
                        continue
                except Exception:
                    # malformed item; skip
                    continue

                try:
                    prev = globals().get("_queued_transcriber", None)
                    try:
                        if transcriber:
                            globals()['_queued_transcriber'] = transcriber
                        # Ensure the async loop + client are started
                        try:
                            if _realtime_async_loop is None:
                                _start_realtime_async_loop()
                        except Exception:
                            pass

                        # Validate WAV bytes before scheduling. If empty or None
                        # then skip and record debug info; this addresses cases
                        # where audio frames may not have been combined correctly
                        # or the processor has no data yet.
                        submit_res = None
                        try:
                            if not wav_bytes:
                                try:
                                    ds = globals().get("_debug_snapshot")
                                    if isinstance(ds, dict):
                                        ds["last_submit_error"] = "empty_wav_bytes"
                                        ds["last_submit_ts"] = time.time()
                                except Exception:
                                    pass
                                submit_res = None
                            else:
                                # Schedule async processing of this item so HTTP I/O and
                                # polling don't block the submitter thread. The async task
                                # appends results into _realtime_results when complete.
                                if _realtime_async_loop is not None:
                                    fut = asyncio.run_coroutine_threadsafe(
                                        _process_realtime_item_async(
                                            wav_bytes,
                                            language,
                                            session_id,
                                            transcriber,
                                            model=model,
                                            transport=transport,
                                        ),
                                        _realtime_async_loop,
                                    )
                                    submit_res = fut
                        except Exception:
                            submit_res = None
                    finally:
                        if prev is None:
                            globals().pop("_queued_transcriber", None)
                        else:
                            globals()['_queued_transcriber'] = prev
                except Exception:
                    submit_res = None

                # record submission result in debug snapshot
                try:
                    ds = globals().get("_debug_snapshot")
                    if isinstance(ds, dict):
                        ds["last_submit_result"] = type(submit_res).__name__ if submit_res is not None else None
                        ds["last_submit_ts"] = time.time()
                except Exception:
                    pass

                # If we received an immediate result dict, append it to results.
                if isinstance(submit_res, dict):
                    try:
                        with _realtime_results_lock:
                            _realtime_results.append(submit_res)
                    except Exception:
                        pass
                else:
                    # Non-dict results (e.g. None) are recorded in debug snapshot only.
                    try:
                        ds = globals().get("_debug_snapshot")
                        if isinstance(ds, dict):
                            ds["last_submit_result_type"] = type(submit_res).__name__ if submit_res is not None else None
                    except Exception:
                        pass
                # continue loop immediately to allow more WAVs to be submitted
            except Exception:
                time.sleep(0.1)

    t1 = threading.Thread(target=_submitter_worker, daemon=True)
    t1.start()
    _realtime_worker_started = True


def _start_debug_http_server(port: int = 8765):
    """Start a tiny HTTP server in a daemon thread to expose the
    module-global `_debug_snapshot` to client-side JS polling.
    """
    from http.server import SimpleHTTPRequestHandler, HTTPServer

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/_debug_snapshot"):
                try:
                    payload = json.dumps(_debug_snapshot)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    # Allow cross-origin requests from Streamlit page
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(payload.encode("utf-8"))
                except Exception:
                    self.send_response(500)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    def _serve():
        try:
            httpd = HTTPServer(("127.0.0.1", port), Handler)
            httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()


def _start_debug_sampler(webrtc_ctx, interval: float = 0.25):
    """Background sampler that updates the module-global _debug_snapshot."""

    def _worker():
        while True:
            # Read the latest webrtc_ctx from a module-global reference so the
            # sampler doesn't call Streamlit APIs from a background thread.
            try:
                local_ctx = (
                    _webrtc_ctx_ref if _webrtc_ctx_ref is not None else webrtc_ctx
                )
            except Exception:
                local_ctx = webrtc_ctx
            try:
                # Safely sample a few simple primitives from live and stored processors
                try:
                    dbg_playing = bool(getattr(local_ctx.state, "playing", False))
                except Exception:
                    dbg_playing = False

                live_proc = getattr(local_ctx, "audio_processor", None)
                stored_proc = _last_audio_proc_ref

                def _frames_len(p):
                    try:
                        return len(getattr(p, "audio_frames", []) or [])
                    except Exception:
                        return 0

                captured_frames_live = _frames_len(live_proc)
                captured_frames_stored = _frames_len(stored_proc)

                recv_cnt = 0
                try:
                    recv_cnt = int(
                        getattr(
                            live_proc,
                            "recv_queued_count",
                            getattr(stored_proc, "recv_queued_count", 0) or 0,
                        )
                    )
                except Exception:
                    recv_cnt = 0

                audio_recv = getattr(local_ctx, "audio_receiver", None)
                a_has_track = False
                a_qsize = None
                try:
                    if audio_recv is not None:
                        a_has_track = bool(audio_recv.hasTrack())
                        q = getattr(audio_recv, "_frames_queue", None)
                        a_qsize = (
                            int(q.qsize())
                            if q is not None and hasattr(q, "qsize")
                            else None
                        )
                except Exception:
                    a_has_track = False

                last_err = None
                try:
                    last_err = getattr(live_proc, "last_error", None) or getattr(
                        stored_proc, "last_error", None
                    )
                except Exception:
                    last_err = None

                snapshot = {
                    "webrtc_playing": bool(dbg_playing),
                    "audio_processor_attached": live_proc is not None,
                    "stored_audio_processor_attached": stored_proc is not None,
                    "audio_processor_is_recording": bool(
                        getattr(live_proc, "is_recording", False)
                        or getattr(stored_proc, "is_recording", False)
                    ),
                    "audio_receiver_attached": audio_recv is not None,
                    "audio_receiver_has_track": bool(a_has_track),
                    "audio_receiver_queue_size": a_qsize,
                    "captured_frames_live": int(captured_frames_live),
                    "captured_frames_stored": int(captured_frames_stored),
                    "recv_queued_count": int(recv_cnt),
                    "last_error": str(last_err) if last_err is not None else None,
                    "ts": time.time(),
                    # diagnostic fields to help debug why counts aren't changing
                    "live_proc_type": type(live_proc).__name__
                    if live_proc is not None
                    else None,
                    "live_proc_id": id(live_proc) if live_proc is not None else None,
                    "stored_proc_type": type(stored_proc).__name__
                    if stored_proc is not None
                    else None,
                    "stored_proc_id": id(stored_proc)
                    if stored_proc is not None
                    else None,
                    "live_proc_has_audio_frames_attr": hasattr(
                        live_proc, "audio_frames"
                    ),
                    "stored_proc_has_audio_frames_attr": hasattr(
                        stored_proc, "audio_frames"
                    ),
                    "live_and_stored_same_object": (live_proc is stored_proc),
                }

                # Merge into module-global snapshot so server flags persist
                try:
                    # use globals() to access module-level variable without 'global'
                    ds = globals().get("_debug_snapshot")
                    if not isinstance(ds, dict):
                        # create a fresh dict in module globals
                        globals()["_debug_snapshot"] = {}
                        ds = globals()["_debug_snapshot"]
                    ds.update(snapshot)
                except Exception:
                    # try to set a module global dict copy of snapshot
                    try:
                        globals()["_debug_snapshot"] = dict(snapshot)
                    except Exception:
                        pass
            except Exception:
                # Keep last exception for diagnosis
                try:
                    ds = globals().get("_debug_snapshot")
                    if not isinstance(ds, dict):
                        globals()["_debug_snapshot"] = {}
                        ds = globals()["_debug_snapshot"]
                    ds["last_exception"] = traceback.format_exc()
                    ds["ts_exception"] = time.time()
                except Exception:
                    pass
            time.sleep(interval)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# WebRTC Configuration
# For local POC we disable STUN to avoid external STUN calls and related
# retry tracebacks. For production / multi-network scenarios enable STUN/TURN.
# Make STUN usage configurable via environment variable ENABLE_STUN.
# For demos in airgapped environments set ENABLE_STUN=0 or false to disable.
ENABLE_STUN = os.getenv("ENABLE_STUN", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Optional: allow providing full ICE servers JSON via RTC_ICE_SERVERS env var.
# Example: RTC_ICE_SERVERS='[{"urls":"stun:stun.l.google.com:19302"}]'
ice_servers: Optional[List[RTCIceServer]] = []

if ENABLE_STUN:
    ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

# Use typed RTCIceServer list (or empty list) when constructing RTCConfiguration
RTC_CONFIGURATION = RTCConfiguration(iceServers=ice_servers)


def audio_processor_factory():
    """Factory for streamlit-webrtc to create AudioProcessor instances."""
    return AudioProcessor()


def _start_debug_thread(webrtc_ctx):
    """Start a background thread that periodically samples debug info and
    stores it in st.session_state['_debug_snapshot'].

    Note: This is a lightweight POC. Writing to Streamlit session_state from
    a background thread can be racey; it's acceptable for local debugging but
    may need rework for production (use proper thread-safe queues or server
    push mechanisms).
    """

    def _worker():
        global _debug_snapshot, _webrtc_ctx_ref, _last_audio_proc_ref, _debug_enabled
        while True:
            try:
                if not _debug_enabled:
                    time.sleep(0.25)
                    continue

                try:
                    local_ctx = (
                        _webrtc_ctx_ref if _webrtc_ctx_ref is not None else webrtc_ctx
                    )
                except Exception:
                    local_ctx = webrtc_ctx

                try:
                    dbg_playing = bool(getattr(local_ctx.state, "playing", False))
                except Exception:
                    dbg_playing = False

                dbg_audio_proc = getattr(local_ctx, "audio_processor", None)
                stored = _last_audio_proc_ref
                dbg_frames = (
                    len(dbg_audio_proc.audio_frames)
                    if dbg_audio_proc
                    and getattr(dbg_audio_proc, "audio_frames", None) is not None
                    else 0
                )
                stored_fr = (
                    len(stored.audio_frames)
                    if stored and getattr(stored, "audio_frames", None) is not None
                    else 0
                )
                dbg_recv = getattr(dbg_audio_proc, "recv_queued_count", 0)
                audio_recv = getattr(local_ctx, "audio_receiver", None)
                a_has_track = False
                a_qsize = None
                try:
                    if audio_recv is not None:
                        a_has_track = audio_recv.hasTrack()
                        q = getattr(audio_recv, "_frames_queue", None)
                        a_qsize = (
                            q.qsize() if q is not None and hasattr(q, "qsize") else None
                        )
                except Exception:
                    a_has_track = False

                snapshot: Dict[str, Any] = {
                    "webrtc_playing": dbg_playing,
                    "audio_processor_attached": dbg_audio_proc is not None,
                    "stored_audio_processor_attached": stored is not None,
                    "audio_processor_is_recording": getattr(
                        dbg_audio_proc, "is_recording", None
                    ),
                    "audio_receiver_attached": audio_recv is not None,
                    "audio_receiver_has_track": a_has_track,
                    "audio_receiver_queue_size": a_qsize,
                    "captured_frames_live": dbg_frames,
                    "captured_frames_stored": stored_fr,
                    "recv_queued_count": dbg_recv,
                    "last_error": getattr(dbg_audio_proc, "last_error", None)
                    or getattr(stored, "last_error", None),
                }

                try:
                    if not isinstance(_debug_snapshot, dict):
                        _debug_snapshot = {}
                    _debug_snapshot.update(snapshot)
                except Exception:
                    try:
                        _debug_snapshot = snapshot
                    except Exception:
                        pass

            except Exception:
                # Avoid thread death on unexpected errors.
                try:
                    _debug_snapshot["last_exception"] = traceback.format_exc()
                except Exception:
                    pass
            time.sleep(0.25)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def transcribe_audio(
    audio_data: bytes, language: Optional[str] = None, transcriber: Optional[str] = None
) -> Optional[dict]:
    """
    Send audio to FastAPI backend for transcription.

    Args:
        audio_data: Audio data in WAV format
        language: Optional language code

    Returns:
        Transcription result
    """
    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    params = {}
    if language:
        params["language"] = language
    if transcriber:
        params["transcriber"] = transcriber

    # If a session is active, use the session-scoped transcription endpoint
    session_id = st.session_state.get("session_id")
    try:
        if session_id:
            try:
                resp = requests.post(
                    f"{API_URL}/sessions/{session_id}/transcribe",
                    files=files,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                jr = resp.json() or {}
                job_id = jr.get("job_id")
                if not job_id:
                    return None

                # Poll the jobs endpoint until done or timeout
                poll_url = f"{API_URL}/sessions/transcribe/jobs/{job_id}"
                deadline = time.time() + 60.0  # 60s overall timeout
                while time.time() < deadline:
                    try:
                        pj = requests.get(poll_url, timeout=10)
                        pj.raise_for_status()
                        job = pj.json() or {}
                        status = job.get("status")
                        if status == "done":
                            return job.get("result")
                        if status == "failed":
                            return None
                    except requests.exceptions.RequestException:
                        # transient error; wait and retry
                        pass
                    time.sleep(0.5)
                # timed out
                return None
            except requests.exceptions.RequestException as e:
                # session POST failed
                try:
                    st.error(f"Error creating transcription job: {str(e)}")
                except Exception:
                    pass
                return None
        else:
            # No session: create a short-lived session and use the
            # session-scoped transcribe endpoint (the backend does not
            # expose a global /transcribe endpoint in some deployments).
            try:
                payload = {
                    "model": st.session_state.get("selected_model"),
                    "language": language,
                    "transcriber": transcriber or st.session_state.get("selected_transcriber"),
                }
                resp = requests.post(f"{API_URL}/sessions", json=payload, timeout=10)
                resp.raise_for_status()
                jr = resp.json() or {}
                sid = jr.get("session_id")
                if not sid:
                    try:
                        st.error("Session endpoint did not return a session_id")
                    except Exception:
                        pass
                    return None

                # Sanity-check: ensure session exists on the server before uploading
                try:
                    sstat = requests.get(f"{API_URL}/sessions/{sid}", timeout=5)
                    if sstat.status_code != 200:
                        txt = None
                        try:
                            txt = sstat.text
                        except Exception:
                            txt = None
                        try:
                            st.error(f"Created session not found on server: {sstat.status_code} - {txt}")
                        except Exception:
                            pass
                        # abort rather than try to upload to a missing session
                        return None
                except requests.exceptions.RequestException:
                    # cannot contact session endpoint; report and abort
                    try:
                        st.error("Failed to verify session on server before upload")
                    except Exception:
                        pass
                    return None

                # Submit file to the created session
                try:
                    r2 = requests.post(
                        f"{API_URL}/sessions/{sid}/transcribe", files=files, timeout=10
                    )
                    # If the server returns non-200, capture body for diagnostic
                    if r2.status_code != 200:
                        body = None
                        try:
                            body = r2.text
                        except Exception:
                            body = None
                        try:
                            st.error(f"Upload failed: {r2.status_code} - {body}")
                        except Exception:
                            pass
                        return None
                    r2.raise_for_status()
                    jr2 = r2.json() or {}
                    job_id = jr2.get("job_id")
                    if not job_id:
                        return None
                except requests.exceptions.RequestException as e:
                    try:
                        # include response text when available for debugging
                        resp_txt = getattr(e, 'response', None)
                        detail = None
                        if resp_txt is not None:
                            try:
                                detail = resp_txt.text
                            except Exception:
                                detail = str(resp_txt)
                        st.error(f"Error creating transcription job: {e} - {detail}")
                    except Exception:
                        pass
                    return None

                poll_url = f"{API_URL}/sessions/transcribe/jobs/{job_id}"
                deadline = time.time() + 60.0
                while time.time() < deadline:
                    try:
                        pj = requests.get(poll_url, timeout=10)
                        pj.raise_for_status()
                        job = pj.json() or {}
                        status = job.get("status")
                        if status == "done":
                            return job.get("result")
                        if status == "failed":
                            return None
                    except requests.exceptions.RequestException:
                        pass
                    time.sleep(0.5)
                return None
            except requests.exceptions.RequestException as e:
                try:
                    st.error(f"Error connecting to API: {str(e)}")
                except Exception:
                    pass
                return None
    except requests.exceptions.RequestException as e:
        try:
            st.error(f"Error connecting to API: {str(e)}")
        except Exception:
            pass
        return None


def transcribe_audio_ws(audio_data: bytes, language: Optional[str] = None, transcriber: Optional[str] = None) -> Optional[dict]:
    """Transcribe audio via backend WebSocket endpoint.

    This creates a session if needed, opens a websocket to /ws/sessions/{session_id},
    sends the WAV bytes, then sends a control 'end' to request final transcription.
    Returns the first 'final' message dict received or None.
    """
    # Ensure we have a session on the server
    session_id = st.session_state.get("session_id")
    sid = session_id
    if not sid:
        payload = {
            "model": st.session_state.get("selected_model"),
            "language": language,
            "transcriber": transcriber or st.session_state.get("selected_transcriber"),
        }
        try:
            resp = requests.post(f"{API_URL}/sessions", json=payload, timeout=10)
            resp.raise_for_status()
            jr = resp.json() or {}
            sid = jr.get("session_id")
            if not sid:
                return None
        except requests.exceptions.RequestException:
            return None

    # Build ws URL from API_URL
    if API_URL.startswith("https://"):
        ws_url = API_URL.replace("https://", "wss://")
    elif API_URL.startswith("http://"):
        ws_url = API_URL.replace("http://", "ws://")
    else:
        ws_url = API_URL
    ws_url = ws_url.rstrip("/") + f"/ws/sessions/{sid}"

    async def _ws_run():
        try:
            async with websockets.connect(ws_url) as ws:
                # send audio as a binary frame
                await ws.send(audio_data)
                # request a finalization immediately
                try:
                    await ws.send(json.dumps({"type": "control", "action": "end"}))
                except Exception:
                    pass

                # wait for final result
                deadline = time.time() + 30.0
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if time.time() > deadline:
                            return None
                        continue
                    if not msg:
                        continue
                    if isinstance(msg, (bytes, bytearray)):
                        continue
                    try:
                        obj = json.loads(msg)
                    except Exception:
                        continue
                    mtype = obj.get("type")
                    if mtype == "final":
                        return obj
                    if mtype == "error":
                        return None
                    if mtype == "done":
                        return None
        except Exception:
            return None

    try:
        return asyncio.run(_ws_run())
    except Exception:
        return None


def _combine_segments_to_wav_bytes(
    segments: List[pydub.AudioSegment],
) -> Optional[bytes]:
    """Combine a list of pydub.AudioSegment into WAV bytes (or None if empty)."""
    if not segments:
        return None
    combined = pydub.AudioSegment.empty()
    for s in segments:
        combined += s
    wav_io = io.BytesIO()
    combined.export(wav_io, format="wav")
    return wav_io.getvalue()


def _is_speech_segments(segments: List[pydub.AudioSegment], dbfs_threshold: float = -40.0, min_duration_s: float = 0.2) -> bool:
    """Rudimentary energy-based VAD over a list of AudioSegment.

    Returns True when combined audio is louder than dbfs_threshold and at
    least min_duration_s long. Thresholds are conservative for web mic
    input; they can be tuned per environment.
    """
    if not segments:
        return False
    try:
        combined = pydub.AudioSegment.empty()
        for s in segments:
            combined += s
        # duration in seconds
        dur = len(combined) / 1000.0
        if dur < float(min_duration_s):
            return False
        # pydub's dBFS can be inf/-inf for silence; guard against that
        try:
            db = combined.dBFS
        except Exception:
            db = float("-inf")
        return db > float(dbfs_threshold)
    except Exception:
        return False

def render_sidebar():     
    with st.sidebar:        
        try:
            status_data = fetch_api_info(API_URL) or {}
            if status_data:
                st.success(f"✅ Device: {status_data.get('device', 'unknown')}")
            else:
                st.error("❌ API Connection Failed")
        except Exception as e:
            status_data = {}
            st.error(f"❌ API Unavailable: {str(e)}")
            st.warning("Make sure the FastAPI backend is running on port 8080")
            
        st.session_state["selected_transcriber"] = "whisper"
        st.session_state["selected_model"] = "tiny.en"
        st.session_state["selected_language"] = "en"
        
        st.divider()
        
        st.checkbox("Show debug info", value=False, key="show_debug_info")
        st.number_input(
            "Debug refresh interval (s)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.get("debug_interval", 1.0),
            step=0.5,
            key="debug_interval",
        )
        
        st.divider()

        
        sel_transport = st.segmented_control(
            "Transport Protocol",
            options=["REST", "WebSocket"],
            selection_mode="single",            
            default=st.session_state.get("realtime_transport", "REST"),
        )

        st.session_state["realtime_transport"] = sel_transport
        
        st.checkbox(
            "Start Realtime Transcribe",
            value=st.session_state.get("realtime_enabled", False),
            key="realtime_enabled",
        )
        try:
            # Interval for chunking (seconds) for realtime mode (sidebar)
            st.number_input(
                "Realtime chunk interval (s)",
                min_value=0.25,
                max_value=5.0,
                value=st.session_state.get("realtime_interval", 1.0),
                step=0.25,
                key="realtime_interval",
            )

            # VAD tuning controls: to adjust energy threshold and silence timeout
            try:
                # slider for dBFS threshold (pydub reports negative values)
                st.slider(
                    "VAD energy threshold (dBFS)",
                    min_value=-70.0,
                    max_value=-10.0,
                    value=st.session_state.get("vad_dbfs_threshold", -40.0),
                    step=1.0,
                    help="Lower (more negative) values make VAD less sensitive; higher values (e.g. -20) are more sensitive to quieter speech.",
                    key="vad_dbfs_threshold",
                )

                st.number_input(
                    "VAD silence timeout (s)",
                    min_value=0.5,
                    max_value=15.0,
                    value=st.session_state.get("vad_silence_secs", 5.0),
                    step=0.5,
                    help="Seconds of continuous silence before finalizing an utterance.",
                    key="vad_silence_secs",
                )
                # Optional: minimum duration for a chunk to be considered speech
                if "vad_min_duration" not in st.session_state:
                    st.session_state["vad_min_duration"] = 0.15

                st.number_input(
                    "VAD min speech duration (s)",
                    min_value=0.05,
                    max_value=1.0,
                    step=0.05,
                    help="Minimum chunk duration to consider as speech (helps ignore very short pops).",
                    key="vad_min_duration",
                )
            except Exception:
                pass
        except Exception:
            pass


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Demo Page", page_icon="🎤", layout="wide")
    st.title("Speech-to-Text Demo")
    selected_language = "en"
    render_sidebar()
      
    try:
        global _debug_enabled
        _debug_enabled = bool(st.session_state.get("show_debug_info", False))
    except Exception:
        _debug_enabled = False

    col1, col2 = st.columns([1, 1])

    with col1:
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = None
        
        if st.session_state.get("session_id"):
            st.info(f"Active session: {st.session_state.get('session_id')}")
            if st.button("End Session", use_container_width=True):                
                st.session_state["session_id"] = None                
                st.session_state["end_webrtc"] = True                
                st.session_state.pop("start_webrtc", None)
                st.success("Session ended")
                st.rerun()                
        else:
            if st.button("Start Session", use_container_width=True):
                payload = {
                    "model": st.session_state.get("selected_model"),
                    "language": selected_language,
                    "transcriber": st.session_state.get("selected_transcriber"),
                }
                
                resp = requests.post(f"{API_URL}/sessions", json=payload, timeout=5)
                resp.raise_for_status()
                data = resp.json() or {}
                sid = data.get("session_id")
                if sid:
                    st.session_state["session_id"] = sid                        
                    st.session_state["start_webrtc"] = True                        
                    st.session_state.pop("end_webrtc", None)
                    st.session_state.pop("last_audio_proc", None)                        
                    st.session_state.pop("last_audio_wav", None)
                    st.session_state["realtime_last_sent"] = 0
                    st.session_state["realtime_transcript"] = []
                    globals()["_webrtc_ctx_ref"] = None
                    globals()["_last_audio_proc_ref"] = None
                    with _realtime_results_lock:
                            _realtime_results.clear()
                    globals()["_realtime_queue"] = Queue(maxsize=8)
                    st.success(f"Session created: {sid}")
                else:
                    st.error("Session endpoint did not return a session_id")
                


                if st.session_state.get("start_webrtc"):
                    js = """
                            <div>
                                <p>To auto-start recording, please allow microphone access.</p>
                                <button id="allow-mic" style="padding:8px 12px; font-size:14px;">Allow microphone & continue</button>
                            </div>
                            <script>
                                const btn = document.getElementById('allow-mic');
                                btn.addEventListener('click', async () => {
                                    try {
                                        await navigator.mediaDevices.getUserMedia({ audio: true });
                                        // reload so server-side can attempt to start the stream
                                        window.location.reload();
                                    } catch (e) {
                                        alert('Microphone permission denied or unavailable: ' + e);
                                    }
                                });
                            </script>
                            """
                    components.html(js, height=120)


        # WebRTC streamer with AudioProcessor attached so recv() is called
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": False, "audio": True},
            audio_receiver_size=1024,
            audio_processor_factory=audio_processor_factory,
            async_processing=True,
        )



        if st.session_state.get("start_webrtc"):
            try:
                playing = bool(getattr(webrtc_ctx.state, "playing", False))
            except Exception:
                playing = False

            try:                
                if not playing:
                    start_fn = getattr(webrtc_ctx, "start", None)
                    if callable(start_fn):
                        try:
                            start_fn()
                        except Exception:                            
                            pass
            except Exception:                
                pass

            
            st.session_state["start_webrtc"] = False


        try:
            if st.session_state.get("end_webrtc"):
                try:
                    playing = bool(getattr(webrtc_ctx.state, "playing", False))
                except Exception:
                    playing = False

                try:
                    stop_fn = getattr(webrtc_ctx, "stop", None)
                    if callable(stop_fn):
                        try:
                            stop_fn()
                        except Exception:
                            pass
                except Exception:
                    pass

                # Try to stop attached audio processor recording if present
                try:
                    audio_proc = getattr(webrtc_ctx, "audio_processor", None)
                    if audio_proc is not None and hasattr(audio_proc, "stop_recording"):
                        try:
                            audio_proc.stop_recording()
                        except Exception:
                            pass
                except Exception:
                    pass

                # Clear end flag and update streaming state
                try:
                    st.session_state["end_webrtc"] = False
                except Exception:
                    pass
                try:
                    st.session_state["is_streaming"] = False
                except Exception:
                    pass
        except Exception:
            pass

        try:
            globals()["_webrtc_ctx_ref"] = webrtc_ctx
        except Exception:
            pass


        try:
            is_playing = bool(getattr(webrtc_ctx.state, "playing", False))
        except Exception:
            is_playing = False

        audio_receiver = getattr(webrtc_ctx, "audio_receiver", None)
        audio_receiver_has_track = False
        try:
            if audio_receiver is not None:
                audio_receiver_has_track = audio_receiver.hasTrack()
        except Exception:
            audio_receiver_has_track = False

        # The checkbox control was moved to the sidebar and stored in
        # session_state under the key 'show_debug_info'. Read that here
        # so the debug output remains in this section of the UI.
        if st.session_state.get("show_debug_info", False):
            # Ensure the tiny debug HTTP server and sampler are started once.
            debug_port = int(os.getenv("DEBUG_HTTP_PORT", "8765"))
            if not st.session_state.get("debug_http_started", False):
                try:
                    _start_debug_http_server(port=debug_port)
                    st.session_state["debug_http_started"] = True
                except Exception:
                    # If starting the local debug HTTP server fails, show a hint
                    st.warning(
                        "Unable to start local debug HTTP server for realtime updates."
                    )

            if not st.session_state.get("debug_sampler_started", False):
                try:
                    # Start sampler which writes to module-global _debug_snapshot
                    _start_debug_sampler(
                        webrtc_ctx,
                        interval=float(st.session_state.get("debug_interval", 1.0)),
                    )
                    st.session_state["debug_sampler_started"] = True
                except Exception:
                    st.warning("Unable to start debug sampler thread.")

            # Render a small client-side polling UI that fetches the snapshot
            # from the local debug HTTP server and updates in-place (no full reload).
            try:
                interval = float(st.session_state.get("debug_interval", 1.0))
            except Exception:
                interval = 1.0

            poll_ms = max(100, int(interval * 1000))
            js_html = f"""<div id="debug-root" style="font-family: monospace; white-space: pre; background:#111; color:#eee; padding:8px; border-radius:6px; height:300px; overflow:auto;"></div>
                            <script>
                                const root = document.getElementById('debug-root');
                                async function fetchSnapshot() {{
                                try {{
                                    const r = await fetch('http://127.0.0.1:{debug_port}/_debug_snapshot');
                                    if (!r.ok) {{
                                    root.textContent = 'Debug endpoint returned ' + r.status;
                                    return;
                                    }}
                                    const j = await r.json();
                                    root.textContent = JSON.stringify(j, null, 2);
                                }} catch (e) {{
                                    root.textContent = 'Fetch error: ' + e;
                                }}
                                }}
                                fetchSnapshot();
                                if (window._debugPoll) clearInterval(window._debugPoll);
                                window._debugPoll = setInterval(fetchSnapshot, {poll_ms});
                            </script>"""

            components.html(js_html, height=340)

        # Recording controls
        col_controls, col_transcribe = st.columns([2, 1])

        with col_controls:
            # Initialize streaming flag in session state
            if "is_streaming" not in st.session_state:
                st.session_state.is_streaming = False

            # Safely read the WebRTC playing state
            try:
                is_playing = bool(getattr(webrtc_ctx.state, "playing", False))
            except Exception:
                is_playing = False

            # Access the audio processor attached to the WebRTC context
            audio_proc = getattr(webrtc_ctx, "audio_processor", None)
            # Persist the last attached audio processor so its captured frames
            # remain available after the WebRTC stream stops.
            if audio_proc is not None:
                st.session_state["last_audio_proc"] = audio_proc
                try:
                    globals()["_last_audio_proc_ref"] = audio_proc
                except Exception:
                    pass
            audio_receiver = getattr(webrtc_ctx, "audio_receiver", None)
            audio_receiver_has_track = False
            try:
                if audio_receiver is not None:
                    audio_receiver_has_track = audio_receiver.hasTrack()
            except Exception:
                audio_receiver_has_track = False

            # Prefer starting recording when an actual audio track is attached to the receiver
            should_record = bool(is_playing or audio_receiver_has_track)

            # Start recording when the WebRTC stream (or receiver track) becomes available
            if should_record and not st.session_state.is_streaming:
                st.session_state.is_streaming = True
                if audio_proc:
                    audio_proc.start_recording()
                    # ensure the session stored reference points to the current processor
                    st.session_state["last_audio_proc"] = audio_proc
                    # Clear any previously stored WAV when a new recording starts
                    if "last_audio_wav" in st.session_state:
                        del st.session_state["last_audio_wav"]
                st.success("Recording started!")

            # Stop recording when both playing and receiver track are gone
            if not should_record and st.session_state.is_streaming:
                st.session_state.is_streaming = False
                # If the live audio processor is attached, stop it and persist
                # the WAV bytes into session_state so Transcribe can be used
                # after the WebRTC stream has been stopped.
                if audio_proc:
                    audio_proc.stop_recording()
                    try:
                        wav = audio_proc.get_audio_data()
                        if wav:
                            st.session_state["last_audio_wav"] = wav
                    except Exception:
                        pass
                else:
                    # If no live processor, try to finalize stored processor
                    stored = st.session_state.get("last_audio_proc")
                    if stored:
                        try:
                            stored.stop_recording()
                            wav = stored.get_audio_data()
                            if wav:
                                st.session_state["last_audio_wav"] = wav
                        except Exception:
                            pass
                        try:
                            globals()["_last_audio_proc_ref"] = stored
                        except Exception:
                            pass
                st.success("Recording stopped!")

            
            # Ensure realtime_state keys exist
            if "realtime_last_sent" not in st.session_state:
                st.session_state["realtime_last_sent"] = 0
            if "realtime_transcript" not in st.session_state:
                st.session_state["realtime_transcript"] = []
            # Voice-activity detection (VAD) state
            if "speech_active" not in st.session_state:
                st.session_state["speech_active"] = False
            if "last_voice_ts" not in st.session_state:
                st.session_state["last_voice_ts"] = 0
            if "speech_start_index" not in st.session_state:
                st.session_state["speech_start_index"] = None
            # VAD thresholds (tunables)
            if "vad_dbfs_threshold" not in st.session_state:
                # default: -40 dBFS (conservative). Adjust if too sensitive/insensitive.
                st.session_state["vad_dbfs_threshold"] = -40.0
            if "vad_silence_secs" not in st.session_state:
                st.session_state["vad_silence_secs"] = 5.0

            # If realtime mode is enabled and we're recording, poll for new frames and send
            if st.session_state.get("realtime_enabled") and st.session_state.get(
                "is_streaming"
            ):
                # Ensure the background worker is running
                try:
                    _start_realtime_worker()
                except Exception:
                    pass
                placeholder_rt = st.empty()
                # Create a placeholder in the right column so partial
                # realtime transcripts can be updated live as results
                # arrive without waiting for a full Streamlit rerun.
                try:
                    col2_rt_placeholder = col2.empty()
                except Exception:
                    col2_rt_placeholder = None
                # Loop while realtime enabled and recording; uses short sleeps to avoid long blocking
                while st.session_state.get("realtime_enabled") and st.session_state.get(
                    "is_streaming"
                ):
                    audio_proc = getattr(webrtc_ctx, "audio_processor", None)
                    # Prefer live processor, then stored processor
                    proc = (
                        audio_proc
                        if audio_proc is not None
                        else st.session_state.get("last_audio_proc")
                    )
                    segments = (
                        getattr(proc, "audio_frames", []) if proc is not None else []
                    )
                    total = len(segments)
                    last = int(st.session_state.get("realtime_last_sent", 0))
                    if total > last:
                        to_send = segments[last:total]
                        # Run simple VAD on the new chunk
                        try:
                            speech_present = _is_speech_segments(
                                to_send,
                                dbfs_threshold=st.session_state.get("vad_dbfs_threshold", -40.0),
                                min_duration_s=0.15,
                            )
                        except Exception:
                            speech_present = False

                        # If we detect speech, mark speech_active and note start index
                        if speech_present:
                            st.session_state["last_voice_ts"] = time.time()
                            if not st.session_state.get("speech_active"):
                                st.session_state["speech_active"] = True
                                # speech_start_index should point to the first frame of this utterance
                                st.session_state["speech_start_index"] = last

                        # If speech is active, continue sending chunks to realtime queue
                        if st.session_state.get("speech_active"):
                            wav_bytes = _combine_segments_to_wav_bytes(to_send)
                            if wav_bytes:
                                try:
                                    _realtime_queue.put_nowait(
                                            (
                                                wav_bytes,
                                                selected_language,
                                                st.session_state.get("session_id"),
                                                st.session_state.get("selected_transcriber"),
                                                st.session_state.get("selected_model"),
                                                st.session_state.get("realtime_transport", "REST"),
                                            )
                                    )
                                except Full:
                                    # Queue full - drop this chunk to avoid growing backlog
                                    pass

                        # If speech was active but we didn't detect voice in this chunk,
                        # check for silence timeout and finalize the utterance.
                        if (
                            st.session_state.get("speech_active")
                            and not speech_present
                        ):
                            last_voice = float(st.session_state.get("last_voice_ts", 0) or 0)
                            silence_limit = float(st.session_state.get("vad_silence_secs", 5.0))
                            if last_voice > 0 and (time.time() - last_voice) >= silence_limit:
                                # Finalize: combine frames from speech_start_index to current total
                                start_idx = st.session_state.get("speech_start_index") or 0
                                end_idx = total
                                try:
                                    utterance_segments = segments[start_idx:end_idx]
                                    final_wav = _combine_segments_to_wav_bytes(utterance_segments)
                                except Exception:
                                    final_wav = None

                                # Advance realtime_last_sent so we don't re-process the same frames
                                st.session_state["realtime_last_sent"] = total

                                # Reset speech state before doing potentially blocking work
                                st.session_state["speech_active"] = False
                                st.session_state["speech_start_index"] = None
                                st.session_state["last_voice_ts"] = 0

                                # If we have audio, synchronously transcribe and set final result
                                if final_wav:
                                    try:
                                        with st.spinner("Finalizing transcription..."):
                                            transport = st.session_state.get("realtime_transport", "REST")
                                            if transport == "WebSocket":
                                                result = transcribe_audio_ws(
                                                    final_wav,
                                                    selected_language,
                                                    st.session_state.get("selected_transcriber"),
                                                )
                                            else:
                                                result = transcribe_audio(
                                                    final_wav,
                                                    selected_language,
                                                    st.session_state.get("selected_transcriber"),
                                                )
                                        if result:
                                            # store transcription and show final text in UI
                                            st.session_state["transcription"] = result
                                            # reset/replace realtime partials with final text
                                            txt = result.get("text") if isinstance(result, dict) else None
                                            if txt:
                                                st.session_state["realtime_transcript"] = [txt]
                                    except Exception:
                                        pass

                        # Update the last sent index for normal flow when not finalizing
                        if not (st.session_state.get("speech_active") is False and not speech_present):
                            st.session_state["realtime_last_sent"] = total

                    placeholder_rt.write(
                        {
                            "realtime_enabled": True,
                            "frames_total": total,
                            "last_sent_index": st.session_state.get(
                                "realtime_last_sent"
                            ),
                            "transcript_segments": st.session_state.get(
                                "realtime_transcript", []
                            ),
                        }
                    )

                    # Drain any completed results from background worker and merge into session_state
                    try:
                        with _realtime_results_lock:
                            results = list(_realtime_results)
                            _realtime_results.clear()
                    except Exception:
                        results = []

                    for result in results:
                        if not result or not isinstance(result, dict):
                            continue
                        text = result.get("text")

                        # Debug: record last partial into module snapshot for troubleshooting
                        try:
                            ds = globals().get("_debug_snapshot")
                            if isinstance(ds, dict):
                                ds["last_partial_text"] = text
                                ds["last_partial_ts"] = time.time()
                        except Exception:
                            pass

                        # Wake-word handling: if enabled and not currently in listen mode,
                        # check partial text for the wake phrase (case-insensitive).
                        try:
                            wake_enabled = bool(st.session_state.get("wake_enabled", False))
                            wake_word = (st.session_state.get("wake_word") or "").strip().lower()
                        except Exception:
                            wake_enabled = False
                            wake_word = ""

                        if wake_enabled and wake_word:
                            # If not already listening, detect wake phrase
                            if not st.session_state.get("wake_listening", False):
                                if text and wake_word in text.lower():
                                    # Enter listening mode
                                    st.session_state["wake_listening"] = True
                                    st.session_state["wake_listen_deadline"] = time.time() + float(
                                        st.session_state.get("wake_listen_timeout", 10.0)
                                    )
                                    # Debug: record wake event
                                    try:
                                        ds = globals().get("_debug_snapshot")
                                        if isinstance(ds, dict):
                                            evs = ds.get("wake_events", [])
                                            evs.append({"ts": time.time(), "partial": text, "wake_word": wake_word})
                                            ds["wake_events"] = evs[-20:]
                                            ds["wake_listening"] = True
                                            ds["wake_listen_deadline"] = st.session_state.get("wake_listen_deadline")
                                    except Exception:
                                        pass
                                    # Clear previous partials and transcription so listening starts fresh
                                    st.session_state["realtime_transcript"] = []
                                    st.session_state["transcription"] = {"text": "", "segments": []}
                                    # Consume this result (do not append the wake phrase itself)
                                    continue
                            else:
                                # refresh deadline whenever speech arrives while listening
                                try:
                                    st.session_state["wake_listen_deadline"] = time.time() + float(
                                        st.session_state.get("wake_listen_timeout", 10.0)
                                    )
                                except Exception:
                                    pass

                                # Debug: update snapshot with refreshed deadline
                                try:
                                    ds = globals().get("_debug_snapshot")
                                    if isinstance(ds, dict):
                                        ds["wake_listening"] = True
                                        ds["wake_listen_deadline"] = st.session_state.get("wake_listen_deadline")
                                except Exception:
                                    pass

                        # If we're in wake_listening mode, we should capture transcripts
                        capture_for_realtime = True
                        if st.session_state.get("wake_enabled") and not st.session_state.get("wake_listening"):
                            
                            capture_for_realtime = False

                        if text and capture_for_realtime:
                            lst = st.session_state.get("realtime_transcript", [])
                            lst.append(text)
                            st.session_state["realtime_transcript"] = lst

                        # Merge result into main transcription structure as before
                        trans = st.session_state.get("transcription", {"text": "", "segments": []})
                        returned_segments = result.get("segments", []) or []
                        if returned_segments:
                            for seg in returned_segments:
                                if "speaker" not in seg:
                                    seg["speaker"] = "Live"
                                trans.setdefault("segments", []).append(seg)
                        else:
                            if text:
                                seg = {
                                    "start": 0.0,
                                    "end": 0.0,
                                    "text": text,
                                    "speaker": "Live",
                                }
                                trans.setdefault("segments", []).append(seg)

                        if text:
                            base_text = trans.get("text", "") or ""
                            trans["text"] = (base_text + " " + text).strip()
                        st.session_state["transcription"] = trans

                    try:
                        if col2_rt_placeholder is not None:
                            rt_list = st.session_state.get("realtime_transcript", [])
                            if rt_list:
                                combined = " ".join([t.strip() for t in rt_list if t])
                                col2_rt_placeholder.markdown("**Realtime (partial)**\n\n" + combined)
                            else:
                                # clear placeholder when no partials
                                col2_rt_placeholder.empty()
                    except Exception:
                        pass

                    # Use the realtime interval from sidebar (session_state key)
                    try:
                        sleep_for = float(
                            st.session_state.get("realtime_interval", 1.0)
                        )
                    except Exception:
                        sleep_for = 1.0
                    time.sleep(float(sleep_for))

        with col_transcribe:
            if st.button("Transcribe", use_container_width=True):
                audio_proc = getattr(webrtc_ctx, "audio_processor", None)

                # Show capture debug info before attempting to get WAV bytes
                if audio_proc is None:
                    st.warning("Audio processor not attached to WebRTC context.")
                else:
                    frames_len = len(getattr(audio_proc, "audio_frames", []))
                    recv_cnt = getattr(audio_proc, "recv_queued_count", 0)
                    st.write(
                        {
                            "audio_processor_attached": True,
                            "audio_frames_len": frames_len,
                            "recv_queued_count": recv_cnt,
                        }
                    )

                    if frames_len > 0:
                        # Show type info for first frame to help debugging
                        try:
                            first_type = type(audio_proc.audio_frames[0]).__name__
                            st.write({"first_frame_type": first_type})
                        except Exception:
                            pass

                # Attempt to produce WAV bytes and provide a download for inspection
                audio_data = None
                # Prefer live processor, then stored WAV in session_state
                if audio_proc is None:
                    audio_data = st.session_state.get("last_audio_wav")
                else:
                    try:
                        audio_data = audio_proc.get_audio_data()
                    except Exception as e:
                        st.error(f"Error creating WAV from captured frames: {e}")

                if audio_data:
                    # Captured audio is retained in-memory for transcription only.
                    # Per configuration we do not offer a WAV download in the UI.

                    with st.spinner("Transcribing..."):
                        transport = st.session_state.get("realtime_transport", "REST")
                        if transport == "WebSocket":
                            result = transcribe_audio_ws(
                                audio_data, selected_language, st.session_state.get("selected_transcriber")
                            )
                        else:
                            result = transcribe_audio(audio_data, selected_language)

                        if result:
                            st.session_state.transcription = result
                            st.success("Transcription complete!")

                            # End the session client-side (clear session and request WebRTC stop)
                            try:
                                st.session_state["session_id"] = None
                                st.session_state["end_webrtc"] = True
                                st.session_state.pop("start_webrtc", None)
                                st.success("Session ended")
                            except Exception:
                                pass

                            # Speak the transcription result using browser TTS (client-side)
                            try:
                                text_to_speak = result.get("text", "") if isinstance(result, dict) else str(result)
                                # Prefer detected language from result, fall back to selected_language
                                detected_lang = None
                                try:
                                    detected_lang = (result.get("language") if isinstance(result, dict) else None) or selected_language
                                except Exception:
                                    detected_lang = selected_language

                                # Use client-side localStorage to ensure playback occurs only once per unique transcription text.
                                if text_to_speak:
                                    try:
                                        tjs = json.dumps(text_to_speak)
                                        ljs = json.dumps(detected_lang) if detected_lang else '"en-US"'
                                        js = f"""<script>
                                        (function(){{
                                            try{{
                                                const key = 'webstt_tts_played_for';
                                                const text = {tjs};
                                                // If we've already played this exact text, skip speaking and just reload UI state
                                                try{{
                                                    if (localStorage.getItem(key) === text) {{
                                                        setTimeout(() => window.location.reload(), 150);
                                                        return;
                                                    }}
                                                }}catch(e){{}}

                                                // Mark as played, then speak
                                                try{{ localStorage.setItem(key, text); }}catch(e){{}}
                                                const utter = new SpeechSynthesisUtterance(text);
                                                try{{ utter.lang = {ljs}; }}catch(e){{}}
                                                utter.onend = function(){{ try{{ window.location.reload(); }}catch(e){{}} }};
                                                if (window.speechSynthesis) {{ window.speechSynthesis.cancel(); window.speechSynthesis.speak(utter); }}
                                            }}catch(e){{console.log('TTS failed', e);}}
                                        }})();
                                        </script>"""
                                        components.html(js, height=1)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error("Transcription failed!")
                else:
                    st.warning(
                        "No audio recorded yet! Check the debug panel for audio_frames_len and recv_queued_count."
                    )

    with col2:
        st.header("Transcription Result")

        # Show realtime partial transcripts (if any)
        if st.session_state.get("realtime_transcript"):
            st.subheader("Realtime (partial)")
            # Show the concatenated partial transcript so far
            rt_list = st.session_state.get("realtime_transcript", [])
            combined = " ".join([t.strip() for t in rt_list if t])
            st.write(combined)            

        if "transcription" in st.session_state and st.session_state.transcription:
            result = st.session_state.transcription

            # Display transcription text
            st.subheader("Text")
            st.write(result.get("text", ""))

            # Display detected language
            detected_lang = result.get("language")
            if detected_lang:
                st.info(f"Detected Language: {detected_lang}")

            # Display segments (if available)
            segments = result.get("segments", [])
            if segments:
                st.subheader("Segments")
                with st.expander("View detailed segments"):
                    for i, segment in enumerate(segments, 1):
                        speaker = (
                            segment.get("speaker")
                            or segment.get("speaker_label")
                            or "Unknown"
                        )
                        st.markdown(
                            f"**{i}.** [{segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s] — **{speaker}**"
                        )
                        st.write(segment.get("text", ""))
        else:
            st.info("Record audio and click 'Transcribe' to see results here.")


if __name__ == "__main__":
    main()
