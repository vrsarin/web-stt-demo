/* Simple signaling and Node-side peer using wrtc

   - Serves static files from ./public
   - WebSocket signaling: client sends JSON messages {type: 'offer', sdp, id}
     server creates RTCPeerConnection, setsRemoteDescription, creates answer
     and sends back {type:'answer', sdp}
   - When audio tracks arrive on the server, we log and expose a hook where
     you can connect TTS/ASR/IVR logic.

   NOTE: This is a minimal scaffold. In production use TURN/STUN and robust
   session management. `wrtc` must be installed and built for your platform.
*/

const path = require('path');
const fs = require('fs');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

// Attempt to import wrtc; if unavailable the server still starts but
// cannot accept media peer connections server-side.
let wrtc = null;
try {
  wrtc = require('wrtc');
} catch (e) {
  console.warn('wrtc module not available. Server-side WebRTC features (SFU) will be disabled.');
}

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const PUBLIC_DIR = path.join(__dirname, '..', 'public');
app.use(express.static(PUBLIC_DIR));

// Simple one-to-many signaling state (mesh POC)
let broadcasterSocketId = null;
const pcs = new Map(); // socketId -> RTCPeerConnection-like placeholder on server (if using wrtc)

io.on('connection', (socket) => {
  console.log('socket connected', socket.id);

  socket.on('broadcaster', () => {
    broadcasterSocketId = socket.id;
    socket.broadcast.emit('broadcaster');
    console.log('Broadcaster registered:', socket.id);
  });

  socket.on('watcher', () => {
    // A watcher wants to watch the broadcaster
    console.log('Watcher joined:', socket.id);
    if (broadcasterSocketId) {
      // notify broadcaster to prepare an offer for this watcher
      io.to(broadcasterSocketId).emit('watcher', socket.id);
    }
  });

  socket.on('offer', ({ id, sdp }) => {
    // Offer from broadcaster targeted at watcher id -> forward to watcher
    console.log('Offer from', socket.id, 'to', id);
    io.to(id).emit('offer', { id: socket.id, sdp });
  });

  socket.on('answer', ({ id, sdp }) => {
    // Answer from watcher targeted at broadcaster id -> forward
    console.log('Answer from', socket.id, 'to', id);
    io.to(id).emit('answer', { id: socket.id, sdp });
  });

  socket.on('candidate', ({ id, candidate }) => {
    // ICE candidate forwarding
    io.to(id).emit('candidate', { id: socket.id, candidate });
  });

  socket.on('disconnect', () => {
    console.log('socket disconnected', socket.id);
    socket.broadcast.emit('disconnectPeer', socket.id);
    if (socket.id === broadcasterSocketId) {
      broadcasterSocketId = null;
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Web IVR server listening on http://localhost:${PORT}`);
  console.log(`Static files served from ${PUBLIC_DIR}`);
});
