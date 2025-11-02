from contextlib import asynccontextmanager
import logging
import os

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import MODEL_NAME, get_languages_list, models_router, get_models_list, transcribers_router
from .rest_sessions import session_router
from .ws_sessions import ws_router

@asynccontextmanager
async def start_api(app):
    get_languages_list()
    get_models_list()
    yield
    

app = FastAPI(title="STT API")

# Configure logging
LOG_LEVEL = os.getenv("API_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("web_stt_demo.api")
logger.setLevel(LOG_LEVEL)

# Add CORS middleware to allow Streamlit to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/info")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

app.include_router(models_router)
app.include_router(session_router)
app.include_router(ws_router)
app.include_router(transcribers_router)