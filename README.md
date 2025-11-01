# web-stt-demo

Speech to Text on Web - A demonstration of browser-based speech recognition with a Flask backend.

## Features

- 🎤 Real-time speech-to-text conversion using Web Speech API
- 🌐 Browser-based (client-side) speech recognition
- 🔄 Backend API for processing transcribed text
- 📱 Responsive design that works on desktop and mobile
- 💯 Confidence score display for each transcription
- 🎨 Modern, clean user interface

## Requirements

- Python 3.7 or higher
- Modern web browser (Chrome, Edge, or Safari recommended)
- Microphone access

## Installation

1. Clone this repository:
```bash
git clone https://github.com/vrsarin/web-stt-demo.git
cd web-stt-demo
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Click the "Start Recording" button and allow microphone access when prompted

4. Start speaking - your speech will be converted to text in real-time

5. Click "Stop Recording" when finished

6. Use the "Clear" button to reset the transcript

## API Endpoints

### `GET /`
Serves the main UI page

### `POST /api/stt`
Receives transcribed text from the frontend

Request body:
```json
{
  "text": "transcribed speech text",
  "confidence": 0.95,
  "timestamp": "2025-11-01T12:00:00Z"
}
```

Response:
```json
{
  "status": "success",
  "received_text": "transcribed speech text",
  "confidence": 0.95,
  "message": "Text received successfully"
}
```

### `GET /api/health`
Health check endpoint

Response:
```json
{
  "status": "healthy"
}
```

## Browser Compatibility

The Web Speech API is supported in:
- ✅ Google Chrome (Desktop & Android)
- ✅ Microsoft Edge
- ✅ Safari (Desktop & iOS)
- ❌ Firefox (limited support)

## How It Works

1. The frontend uses the browser's built-in Web Speech API (`SpeechRecognition`)
2. When the user clicks "Start Recording", the browser requests microphone access
3. Speech is converted to text in real-time by the browser
4. Interim results (partial transcriptions) are shown in gray
5. Final results are shown in bold black text
6. Each final transcription is sent to the Flask backend via POST request
7. The backend can process, store, or forward the transcribed text

## Development

The project structure:
```
web-stt-demo/
├── app.py              # Flask backend server
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main UI template
└── static/
    ├── style.css      # CSS styling
    └── script.js      # JavaScript for speech recognition
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
