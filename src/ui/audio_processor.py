import io
from typing import List, Optional

import av
import numpy as np
import pydub
from streamlit_webrtc import AudioProcessorBase


class AudioProcessor(AudioProcessorBase):
    """Process audio frames from WebRTC stream."""

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_frames = []
        self.is_recording = False
        self.last_error = None
        self.recv_queued_count = 0

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
            try:
                arr = frame.to_ndarray()

                # Determine channel count from frame layout (reliable)
                try:
                    channels = len(frame.layout.channels) if getattr(frame, "layout", None) is not None else 1
                except Exception:
                    channels = 1
                # Defensive: ensure at least 1 channel
                channels = max(1, int(channels))

                # Convert arr to an interleaved 1-D array of samples
                if arr.ndim == 1:
                    interleaved = arr.flatten()
                elif arr.ndim == 2:
                    # Planar format: (channels, samples)
                    if arr.shape[0] == channels:
                        interleaved = arr.T.reshape(-1)
                    # Interleaved format: (samples, channels)
                    elif arr.shape[1] == channels:
                        interleaved = arr.reshape(-1)
                    else:
                        # Fallback: flatten whatever we have
                        interleaved = arr.flatten()
                else:
                    interleaved = arr.flatten()

                # Convert float audio to int16 PCM if needed
                if np.issubdtype(interleaved.dtype, np.floating):
                    pcm = (interleaved * 32767).astype(np.int16)
                else:
                    if interleaved.dtype.itemsize > 2 and np.issubdtype(interleaved.dtype, np.integer):
                        shift = 8 * (interleaved.dtype.itemsize - 2)
                        pcm = (interleaved // (2 ** shift)).astype(np.int16)
                    else:
                        pcm = interleaved.astype(np.int16)

                raw_bytes = pcm.tobytes()

                sound = pydub.AudioSegment(
                    data=raw_bytes,
                    sample_width=2,
                    frame_rate=frame.sample_rate,
                    channels=channels,
                )
                self.audio_frames.append(sound)
            except Exception as e:
                # Record the error for debugging but don't crash recv
                self.last_error = str(e)

        return frame

    async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        """Process queued audio frames to avoid dropping frames in async mode."""
        for frame in frames:
            if self.is_recording:
                try:

                    arr = frame.to_ndarray()

                    try:
                        channels = len(frame.layout.channels) if getattr(frame, "layout", None) is not None else 1
                    except Exception:
                        channels = 1
                    channels = max(1, int(channels))

                    if arr.ndim == 1:
                        interleaved = arr.flatten()
                    elif arr.ndim == 2:
                        if arr.shape[0] == channels:
                            interleaved = arr.T.reshape(-1)
                        elif arr.shape[1] == channels:
                            interleaved = arr.reshape(-1)
                        else:
                            interleaved = arr.flatten()
                    else:
                        interleaved = arr.flatten()

                    if np.issubdtype(interleaved.dtype, np.floating):
                        pcm = (interleaved * 32767).astype(np.int16)
                    else:
                        if interleaved.dtype.itemsize > 2 and np.issubdtype(interleaved.dtype, np.integer):
                            shift = 8 * (interleaved.dtype.itemsize - 2)
                            pcm = (interleaved // (2 ** shift)).astype(np.int16)
                        else:
                            pcm = interleaved.astype(np.int16)

                    raw_bytes = pcm.tobytes()

                    sound = pydub.AudioSegment(
                        data=raw_bytes,
                        sample_width=2,
                        frame_rate=frame.sample_rate,
                        channels=channels,
                    )
                    self.audio_frames.append(sound)
                    # Increment counter so we can show frames processed in debug UI
                    try:
                        self.recv_queued_count += 1
                    except Exception:
                        pass
                except Exception as e:
                    self.last_error = str(e)

        # Return the frames unchanged (pass-through)
        return list(frames)

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