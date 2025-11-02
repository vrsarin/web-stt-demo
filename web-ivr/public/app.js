(function () {
  const logs = document.getElementById('logs');
  function log(...args) {
    logs.textContent += args
      .map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a)))
      .join(' ') + '\n';
    logs.scrollTop = logs.scrollHeight;
  }

  const pcs = {}; // peerId -> RTCPeerConnection
  let localStream = null;
  let socket = null;

  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const serverUrlInput = document.getElementById('serverUrl');

  function getSelectedRole() {
    const r = document.querySelector('input[name="role"]:checked');
    return r ? r.value : 'viewer';
  }

  async function start() {
    const serverUrl = serverUrlInput.value.trim();
    if (!serverUrl) return alert('Enter signaling URL');

    socket = io(serverUrl);

    socket.on('connect', () => log('socket connected', socket.id));
    socket.on('broadcaster', () => {
      log('broadcaster available');
      // viewers may choose to join when broadcaster announced
    });

    socket.on('watcher', async (id) => {
      // Only broadcaster handles this: create a peer for the watcher
      log('watcher event for', id);
      const pc = new RTCPeerConnection({ iceServers: [] });
      pcs[id] = pc;

      pc.onicecandidate = (ev) => {
        if (ev.candidate) {
          socket.emit('candidate', { id, candidate: ev.candidate });
        }
      };

      // No ontrack for broadcaster (we send audio), but keep for completeness
      pc.ontrack = (ev) => log('broadcaster received track (unexpected)', ev.track.kind);

      // add local audio tracks
      try {
        localStream.getAudioTracks().forEach((t) => pc.addTrack(t, localStream));
      } catch (e) {
        log('Error adding local tracks', e);
      }

      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        socket.emit('offer', { id, sdp: pc.localDescription.sdp });
        log('sent offer to watcher', id);
      } catch (e) {
        log('createOffer failed', e);
      }
    });

    socket.on('offer', async ({ id, sdp }) => {
      // viewer receives offer from broadcaster
      log('offer from', id);
      const pc = new RTCPeerConnection({ iceServers: [] });
      pcs[id] = pc;

      pc.onicecandidate = (ev) => {
        if (ev.candidate) {
          socket.emit('candidate', { id, candidate: ev.candidate });
        }
      };

      pc.ontrack = (ev) => {
        log('received track from broadcaster', ev.track.kind);
        // play the received audio
        const audioEl = document.createElement('audio');
        audioEl.autoplay = true;
        audioEl.srcObject = ev.streams[0];
        document.body.appendChild(audioEl);
      };

      try {
        await pc.setRemoteDescription({ type: 'offer', sdp });
        // create answer
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        socket.emit('answer', { id, sdp: pc.localDescription.sdp });
        log('sent answer to', id);
      } catch (e) {
        log('Error handling offer', e);
      }
    });

    socket.on('answer', async ({ id, sdp }) => {
      log('answer from', id);
      const pc = pcs[id];
      if (pc) {
        try {
          await pc.setRemoteDescription({ type: 'answer', sdp });
        } catch (e) {
          log('setRemoteDescription(answer) failed', e);
        }
      }
    });

    socket.on('candidate', async ({ id, candidate }) => {
      const pc = pcs[id];
      if (pc) {
        try {
          await pc.addIceCandidate(candidate);
        } catch (e) {
          log('addIceCandidate failed', e);
        }
      }
    });

    socket.on('disconnectPeer', (id) => {
      log('peer disconnected', id);
      const pc = pcs[id];
      if (pc) {
        try { pc.close(); } catch (e) {}
        delete pcs[id];
      }
    });

    // start local media if broadcasting
    const role = getSelectedRole();
    if (role === 'broadcaster') {
      try {
        localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        log('got local stream');
      } catch (e) {
        log('getUserMedia failed', e);
        return;
      }

      socket.emit('broadcaster');
    } else {
      // viewer
      socket.emit('watcher');
    }

    startBtn.disabled = true;
    stopBtn.disabled = false;
  }

  function stop() {
    for (const id of Object.keys(pcs)) {
      try { pcs[id].close(); } catch (e) {}
      delete pcs[id];
    }
    if (localStream) {
      localStream.getTracks().forEach((t) => t.stop());
      localStream = null;
    }
    if (socket) {
      try { socket.disconnect(); } catch (e) {}
      socket = null;
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    log('stopped');
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
})();
