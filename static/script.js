// Speech recognition setup
let recognition = null;
let isRecording = false;
let finalTranscript = '';
let interimTranscript = '';

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const transcriptDiv = document.getElementById('transcript');
const statusDiv = document.getElementById('status');
const confidenceDiv = document.getElementById('confidence');

// Check if browser supports Web Speech API
function checkBrowserSupport() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Your browser does not support the Web Speech API. Please use Chrome, Edge, or Safari.');
        startBtn.disabled = true;
        return false;
    }
    return true;
}

// Initialize speech recognition
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    // Configure recognition
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;
    
    // Event handlers
    recognition.onstart = () => {
        console.log('Speech recognition started');
        isRecording = true;
        updateUI();
        statusDiv.textContent = 'Listening...';
        statusDiv.classList.add('recording');
    };
    
    recognition.onresult = (event) => {
        interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            const confidence = event.results[i][0].confidence;
            
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
                
                // Send to backend API
                sendToAPI(transcript, confidence);
                
                // Update confidence display
                if (confidence) {
                    const confidencePercent = (confidence * 100).toFixed(1);
                    confidenceDiv.textContent = `Confidence: ${confidencePercent}%`;
                }
            } else {
                interimTranscript += transcript;
            }
        }
        
        updateTranscript();
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        let errorMessage = 'Error: ';
        
        switch(event.error) {
            case 'no-speech':
                errorMessage += 'No speech detected. Please try again.';
                break;
            case 'audio-capture':
                errorMessage += 'No microphone found. Please check your setup.';
                break;
            case 'not-allowed':
                errorMessage += 'Microphone permission denied.';
                break;
            default:
                errorMessage += event.error;
        }
        
        statusDiv.textContent = errorMessage;
        statusDiv.classList.remove('recording');
        isRecording = false;
        updateUI();
    };
    
    recognition.onend = () => {
        console.log('Speech recognition ended');
        isRecording = false;
        statusDiv.textContent = 'Stopped';
        statusDiv.classList.remove('recording');
        updateUI();
    };
}

// Update transcript display
function updateTranscript() {
    if (finalTranscript === '' && interimTranscript === '') {
        transcriptDiv.innerHTML = '<div class="placeholder">Your speech will appear here...</div>';
    } else {
        let html = '';
        if (finalTranscript) {
            html += `<span class="final">${finalTranscript}</span>`;
        }
        if (interimTranscript) {
            html += `<span class="interim">${interimTranscript}</span>`;
        }
        transcriptDiv.innerHTML = html;
    }
}

// Update UI based on recording state
function updateUI() {
    if (isRecording) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// Send transcribed text to API
async function sendToAPI(text, confidence) {
    try {
        const response = await fetch('/api/stt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                confidence: confidence,
                timestamp: new Date().toISOString()
            })
        });
        
        const data = await response.json();
        console.log('API Response:', data);
    } catch (error) {
        console.error('Error sending to API:', error);
    }
}

// Event listeners
startBtn.addEventListener('click', () => {
    if (!recognition) {
        initSpeechRecognition();
    }
    
    try {
        recognition.start();
        statusDiv.textContent = 'Starting...';
    } catch (error) {
        console.error('Error starting recognition:', error);
    }
});

stopBtn.addEventListener('click', () => {
    if (recognition && isRecording) {
        recognition.stop();
    }
});

clearBtn.addEventListener('click', () => {
    finalTranscript = '';
    interimTranscript = '';
    updateTranscript();
    confidenceDiv.textContent = '';
    statusDiv.textContent = 'Ready';
    statusDiv.classList.remove('recording');
});

// Initialize
if (checkBrowserSupport()) {
    console.log('Web Speech API is supported');
    statusDiv.textContent = 'Ready';
} else {
    statusDiv.textContent = 'Browser not supported';
}
