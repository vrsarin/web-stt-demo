# web-ivr

Minimal Node.js scaffold for a WebRTC-based IVR frontend.

What this provides

- Express static server serving a client that captures microphone and establishes a WebRTC PeerConnection
- A simple WebSocket signaling server
- Use of `wrtc` on the server to accept the incoming peer and expose hooks for handling audio tracks

Quick start

1. cd web-ivr
2. npm install
3. npm start
4. Open http://localhost:3000 in a browser and click "Start". Choose the role `Broadcaster` to stream your microphone, or `Viewer` to receive the broadcast.

Notes and next steps

- This is a scaffold. For a production IVR you'll want to:
  - Add TURN/STUN servers
  - Replace simple in-memory peers map with persistent sessions
  - Implement server-side audio processing (ASR, IVR state, TTS)
  - Harden signaling and add authentication

Server entry: `src/server.js`.
Client entry: `public/index.html` and `public/app.js`.
