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

## Configuration

### Whisper Model Selection

You can choose different Whisper models based on your needs:

- **tiny**: Fastest, least accurate
- **base**: Good balance (default)
- **small**: Better accuracy
- **medium**: High accuracy
- **large**: Best accuracy, slowest

## Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)**: WebRTC component for Streamlit
- **[OpenAI Whisper](https://github.com/openai/whisper)**: Speech recognition model
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern web framework for building APIs
- **[PyTorch](https://pytorch.org/)**: Machine learning framework
- **[PyAV](https://github.com/PyAV-Org/PyAV)**: Python bindings for FFmpeg
- **[Pydub](https://github.com/jiaaro/pydub)**: Audio processing library

