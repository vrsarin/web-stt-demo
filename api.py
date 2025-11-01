"""
FastAPI backend for audio transcription using OpenAI Whisper.
"""
import os
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch

app = FastAPI(title="Whisper STT API")

# Add CORS middleware to allow Streamlit to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")


def load_whisper_model(model_name: str = MODEL_NAME):
    """Load the Whisper model."""
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{model_name}' on {device}...")
        model = whisper.load_model(model_name, device=device)
        print("Model loaded successfully!")
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_whisper_model()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Transcribe audio file using Whisper.
    
    Args:
        file: Audio file (wav, mp3, m4a, etc.)
        language: Optional language code (e.g., 'en', 'es')
    
    Returns:
        Transcription result with text and metadata
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Ensure model is loaded
        whisper_model = load_whisper_model()
        
        # Transcribe
        result = whisper_model.transcribe(
            tmp_file_path,
            language=language,
            fp16=torch.cuda.is_available()
        )
        
        return {
            "text": result["text"],
            "language": result.get("language"),
            "segments": result.get("segments", [])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
