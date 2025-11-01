"""
Streamlit Web Speech-to-Text Demo using WebRTC and OpenAI Whisper.
"""
import os
import io
import tempfile
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import pydub
import requests
from typing import Optional

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
SAMPLE_RATE = 16000

# WebRTC Configuration (using public STUN servers)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class AudioProcessor:
    """Process audio frames from WebRTC stream."""
    
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
    
    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.audio_frames = []
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
    
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Receive and process audio frame."""
        if self.is_recording:
            # Convert frame to numpy array
            sound = pydub.AudioSegment(
                data=frame.to_ndarray().tobytes(),
                sample_width=frame.format.bytes,
                frame_rate=frame.sample_rate,
                channels=len(frame.layout.channels),
            )
            self.audio_frames.append(sound)
        
        return frame
    
    def get_audio_data(self) -> Optional[bytes]:
        """Get combined audio data as WAV bytes."""
        if not self.audio_frames:
            return None
        
        # Combine all audio frames
        combined = pydub.AudioSegment.empty()
        for frame in self.audio_frames:
            combined += frame
        
        # Export as WAV
        wav_io = io.BytesIO()
        combined.export(wav_io, format="wav")
        return wav_io.getvalue()


def transcribe_audio(audio_data: bytes, language: Optional[str] = None) -> dict:
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
    
    try:
        response = requests.post(
            f"{API_URL}/transcribe",
            files=files,
            params=params,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Web Speech-to-Text Demo",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Web Speech-to-Text Demo")
    st.markdown("### Real-time audio transcription using WebRTC and OpenAI Whisper")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API status check
        try:
            response = requests.get(f"{API_URL}/", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                st.success("✅ API Connected")
                st.info(f"Model: {status_data.get('model', 'unknown')}")
                st.info(f"Device: {status_data.get('device', 'unknown')}")
            else:
                st.error("❌ API Connection Failed")
        except Exception as e:
            st.error(f"❌ API Unavailable: {str(e)}")
            st.warning("Make sure the FastAPI backend is running on port 8000")
        
        st.divider()
        
        # Language selection
        language = st.selectbox(
            "Language (optional)",
            options=["Auto-detect", "English", "Spanish", "French", "German", "Chinese"],
            index=0
        )
        
        language_codes = {
            "Auto-detect": None,
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Chinese": "zh"
        }
        selected_language = language_codes[language]
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This demo uses:
        - **Streamlit** for the web interface
        - **WebRTC** for real-time audio streaming
        - **OpenAI Whisper** for speech recognition
        - **FastAPI** for the backend API
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎙️ Audio Input")
        
        # Initialize audio processor in session state
        if "audio_processor" not in st.session_state:
            st.session_state.audio_processor = AudioProcessor()
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": False, "audio": True},
            audio_receiver_size=1024,
            async_processing=True,
        )
        
        st.markdown("---")
        
        # Recording controls
        col_start, col_stop, col_transcribe = st.columns(3)
        
        with col_start:
            if st.button("🔴 Start Recording", use_container_width=True):
                st.session_state.audio_processor.start_recording()
                st.success("Recording started!")
        
        with col_stop:
            if st.button("⏹️ Stop Recording", use_container_width=True):
                st.session_state.audio_processor.stop_recording()
                st.success("Recording stopped!")
        
        with col_transcribe:
            if st.button("📝 Transcribe", use_container_width=True):
                audio_data = st.session_state.audio_processor.get_audio_data()
                
                if audio_data:
                    with st.spinner("Transcribing..."):
                        result = transcribe_audio(audio_data, selected_language)
                        
                        if result:
                            st.session_state.transcription = result
                            st.success("Transcription complete!")
                        else:
                            st.error("Transcription failed!")
                else:
                    st.warning("No audio recorded yet!")
    
    with col2:
        st.header("📄 Transcription Result")
        
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
                        st.markdown(f"**{i}.** [{segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s]")
                        st.write(segment.get("text", ""))
        else:
            st.info("Record audio and click 'Transcribe' to see results here.")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Allow microphone access** when prompted by your browser
        2. **Click 'Start Recording'** to begin capturing audio
        3. **Speak into your microphone**
        4. **Click 'Stop Recording'** when finished
        5. **Click 'Transcribe'** to convert speech to text
        6. View the transcription result on the right panel
        
        **Note:** Make sure the FastAPI backend is running on port 8000.
        You can start it with: `python api.py` or `uvicorn api:app --reload`
        """)


if __name__ == "__main__":
    main()
