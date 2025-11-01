# Web Speech-to-Text Demo

A real-time speech-to-text application built with Streamlit, WebRTC, OpenAI Whisper, and FastAPI.

## Features

- 🎤 **Real-time audio capture** using WebRTC protocol
- 🗣️ **Accurate transcription** powered by OpenAI Whisper
- 🌐 **Web-based interface** built with Streamlit
- ⚡ **Fast API backend** for efficient audio processing
- 🌍 **Multi-language support** with auto-detection
- 📊 **Detailed segment information** with timestamps

## Architecture

- **Frontend**: Streamlit with streamlit-webrtc for audio streaming
- **Backend**: FastAPI server running OpenAI Whisper model
- **Protocol**: WebRTC for real-time audio communication
- **AI Model**: OpenAI Whisper for speech recognition

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Microphone access in your browser

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vrsarin/web-stt-demo.git
cd web-stt-demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Configure environment variables:
```bash
cp .env.example .env
# Edit .env to customize settings
```

## Usage

### Starting the Application

You need to run both the FastAPI backend and Streamlit frontend:

#### Option 1: Using separate terminals

**Terminal 1 - Start the FastAPI backend:**
```bash
python api.py
```
Or with uvicorn:
```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

**Terminal 2 - Start the Streamlit frontend:**
```bash
streamlit run app.py
```

The web interface will open automatically at `http://localhost:8501`

#### Option 2: Using a process manager

You can use a process manager like `honcho` or create a simple script to run both services.

### Using the Application

1. Open the Streamlit web interface in your browser
2. Allow microphone access when prompted
3. Select your preferred language (or use auto-detect)
4. Click **"Start Recording"** to begin capturing audio
5. Speak into your microphone
6. Click **"Stop Recording"** when finished
7. Click **"Transcribe"** to convert speech to text
8. View the transcription result on the right panel

## Configuration

### Whisper Model Selection

You can choose different Whisper models based on your needs:

- **tiny**: Fastest, least accurate
- **base**: Good balance (default)
- **small**: Better accuracy
- **medium**: High accuracy
- **large**: Best accuracy, slowest

Set the model in `.env`:
```
WHISPER_MODEL=base
```

### Supported Languages

The application supports all languages available in Whisper, including:
- English
- Spanish
- French
- German
- Chinese
- And many more...

## API Endpoints

### `GET /`
Health check endpoint that returns API status and configuration.

### `POST /transcribe`
Transcribe audio file to text.

**Parameters:**
- `file`: Audio file (required)
- `language`: Language code (optional, e.g., 'en', 'es')

**Response:**
```json
{
  "text": "Transcribed text",
  "language": "en",
  "segments": [...]
}
```

## Project Structure

```
web-stt-demo/
├── app.py              # Streamlit frontend application
├── api.py              # FastAPI backend server
├── requirements.txt    # Python dependencies
├── .env.example        # Environment configuration template
├── .gitignore         # Git ignore rules
├── LICENSE            # License file
└── README.md          # This file
```

## Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)**: WebRTC component for Streamlit
- **[OpenAI Whisper](https://github.com/openai/whisper)**: Speech recognition model
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern web framework for building APIs
- **[PyTorch](https://pytorch.org/)**: Machine learning framework
- **[PyAV](https://github.com/PyAV-Org/PyAV)**: Python bindings for FFmpeg
- **[Pydub](https://github.com/jiaaro/pydub)**: Audio processing library

## Troubleshooting

### API Connection Failed
- Make sure the FastAPI backend is running on port 8000
- Check that no firewall is blocking the connection
- Verify the API_URL in your environment configuration

### Microphone Not Working
- Ensure your browser has permission to access the microphone
- Check that your microphone is properly connected
- Try refreshing the page and granting permissions again

### Slow Transcription
- Consider using a smaller Whisper model (e.g., 'tiny' or 'base')
- If you have a CUDA-capable GPU, make sure PyTorch is installed with CUDA support

### Memory Issues
- Use a smaller Whisper model
- Reduce the length of audio recordings
- Close other applications to free up memory

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for the Whisper model
- Streamlit team for the amazing framework
- streamlit-webrtc contributors for WebRTC integration
