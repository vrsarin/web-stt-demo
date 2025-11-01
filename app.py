"""
Flask web application for Speech-to-Text demo.
Serves the frontend UI and provides API endpoints.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')

@app.route('/api/stt', methods=['POST'])
def speech_to_text():
    """
    API endpoint to receive speech-to-text results.
    In this demo, the actual STT processing happens client-side using Web Speech API.
    This endpoint can be used to save/process the transcribed text.
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # Validate input
    if not isinstance(text, str):
        return jsonify({'error': 'Text must be a string'}), 400
    
    if len(text) > 10000:  # Reasonable limit for text length
        return jsonify({'error': 'Text exceeds maximum length of 10000 characters'}), 400
    
    confidence = data.get('confidence', None)
    
    # Validate confidence if provided
    if confidence is not None and (not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1):
        return jsonify({'error': 'Confidence must be a number between 0 and 1'}), 400
    
    # Here you could save to database, process the text, etc.
    # For this demo, we'll just return a confirmation
    response = {
        'status': 'success',
        'received_text': text,
        'confidence': confidence,
        'message': 'Text received successfully'
    }
    
    return jsonify(response), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
