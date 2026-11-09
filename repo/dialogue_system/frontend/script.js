const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const voiceCircle = document.getElementById('voiceCircle');
const orbWrap = document.getElementById('orbWrap');
const circleStatusEl = document.getElementById('circleStatus');
const waveformCanvas = document.getElementById('waveformCanvas');
const ctx = waveformCanvas.getContext('2d');
const sessionTimerEl = document.getElementById('sessionTimer');

// Canvas backing store follows the CSS box (device-pixel aware); the layout
// is now full-width, so a resize must re-fit instead of measuring once.
function fitCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = waveformCanvas.getBoundingClientRect();
  waveformCanvas.width = Math.max(1, Math.round(rect.width * dpr));
  waveformCanvas.height = Math.max(1, Math.round(rect.height * dpr));
}
fitCanvas();
window.addEventListener('resize', fitCanvas);

let audioContext, analyser, dataArray, source, stream, processor;
let animationId = null;
let listening = false;

// --- ASR Audio Context is separate from TTS Audio Context ---
let ttsContext = null;
let ttsNode = null;
let ttsSab = null;
let ttsFloat32Data = null;
let ttsStates = null; // Int32Array [writeIndex, readIndex]
// Buffer size: capacity for ~120 seconds at 24k
const TTS_BUFFER_SIZE = 24000 * 120;

// ==================== Session timer ====================
let sessionStart = null;
let timerId = null;

function fmtDuration(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

function startTimer() {
  sessionStart = Date.now();
  sessionTimerEl.textContent = '00:00';
  timerId = setInterval(() => {
    sessionTimerEl.textContent = fmtDuration(Date.now() - sessionStart);
  }, 1000);
}

function stopTimer() {
  if (timerId) { clearInterval(timerId); timerId = null; }
  sessionStart = null;
  sessionTimerEl.textContent = '00:00';
}

// ==================== Native WebSocket (with auto-reconnect) ====================
// Determine protocol (ws or wss)
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws`;

// Reconnect policy: exponential backoff, capped at 5s. A dead socket must
// never leave the page silent: every feature (transcription, VAD status,
// TTS audio) rides on this one connection.
const HEARTBEAT_INTERVAL_MS = 8000;  // app-level ping cadence
const HEARTBEAT_STALE_MS = 25000;    // no server traffic for this long => force reconnect

let socket = null;
let reconnectAttempts = 0;
let reconnectTimer = null;
let heartbeatTimer = null;
let lastServerMsgTime = Date.now();
let loadingConfirmedOnce = false;

function setConnState(state, label) {
  const pill = document.getElementById('connPill');
  const dot = document.getElementById('brandDot');
  if (pill) { pill.dataset.state = state; pill.textContent = label; }
  if (dot) { dot.dataset.state = state; }
}

function connectSocket() {
  setConnState('connecting', 'Connecting…');
  socket = new WebSocket(wsUrl);
  socket.binaryType = 'arraybuffer'; // Important for receiving audio

  socket.onopen = () => {
    console.log('[WebSocket] connected');
    reconnectAttempts = 0;
    lastServerMsgTime = Date.now();
    hideConnBanner();
    setConnState('online', 'Connected');
    startHeartbeat();
    // Session state lives on the server per-connection: re-announce the
    // audio format and drop stale TTS playback from the previous session.
    if (listening && audioContext) {
      socket.send(JSON.stringify({
        event: 'config_audio',
        data: { sample_rate: audioContext.sampleRate }
      }));
    }
    if (ttsNode) ttsNode.port.postMessage({ type: 'abort' });
    // A reconnect means a brand-new server session: settings (persona, TTS
    // params...) were reset server-side. Re-apply whatever the UI holds.
    sendCurrentSettings();
  };

  socket.onclose = () => {
    console.log('[WebSocket] disconnected');
    stopHeartbeat();
    setConnState('offline', 'Reconnecting…');
    scheduleReconnect();
  };

  socket.onerror = (error) => {
    // onclose always follows; reconnection is handled there
    console.error('[WebSocket] error:', error);
  };

  socket.onmessage = onSocketMessage;
}

function scheduleReconnect() {
  showConnBanner();
  if (reconnectTimer) return;
  const delay = Math.min(5000, 500 * Math.pow(2, reconnectAttempts++));
  console.log(`[WebSocket] reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectSocket();
  }, delay);
}

// Skip the backoff wait and connect immediately (manual button, or the
// network just came back — the pending timer, if any, is cancelled).
function reconnectNow() {
  if (socket && (socket.readyState === WebSocket.OPEN ||
                 socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  connectSocket();
}

// The always-visible header button: unlike reconnectNow(), this works even
// when a socket is still nominally OPEN (half-dead TCP, wedged session...).
// The old socket's onclose is detached so it can't schedule a second
// reconnect after we've already started the replacement connection.
function manualReconnect() {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (socket) {
    socket.onclose = null;
    socket.onerror = null;
    try { socket.close(); } catch (e) { /* already dead */ }
  }
  connectSocket();
}

function startHeartbeat() {
  stopHeartbeat();
  heartbeatTimer = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ event: 'ping' }));
    }
    if (Date.now() - lastServerMsgTime > HEARTBEAT_STALE_MS) {
      // Half-dead TCP: send() succeeds but nothing ever comes back, and the
      // browser keeps readyState OPEN. Force-close so onclose can reconnect.
      console.warn('[WebSocket] no server traffic for ' +
        (HEARTBEAT_STALE_MS / 1000) + 's, forcing reconnect');
      try { socket.close(); } catch (e) { /* already closing */ }
    }
  }, HEARTBEAT_INTERVAL_MS);
}

function stopHeartbeat() {
  if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
}

function showConnBanner() {
  const el = document.getElementById('connBanner');
  if (el) el.classList.remove('hidden');
}

function hideConnBanner() {
  const el = document.getElementById('connBanner');
  if (el) el.classList.add('hidden');
}

connectSocket();

// Manual reconnect button on the connection banner.
document.getElementById('reconnectNowBtn').addEventListener('click', reconnectNow);

// Always-visible reconnect button in the top bar.
document.getElementById('reconnectBtn').addEventListener('click', manualReconnect);

// Network-level events: when the OS reports the network is back, don't sit
// out the remaining backoff — the socket is almost certainly dead and the
// retry would succeed right away.
window.addEventListener('online', () => {
  console.log('[WebSocket] network online — attempting immediate reconnect');
  reconnectNow();
});
window.addEventListener('offline', () => {
  console.log('[WebSocket] network offline');
  showConnBanner();
  setConnState('offline', 'Offline');
  // The socket won't notice the drop until a send fails; force the issue
  // so reconnection starts (and completes) the moment we're back.
  try { if (socket) socket.close(); } catch (e) { /* already closing */ }
});
// Returning to a backgrounded tab: timers were throttled, so the heartbeat
// may not have fired for a long time — check staleness right away instead
// of waiting for the next (throttled) tick.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && socket &&
      socket.readyState === WebSocket.OPEN &&
      Date.now() - lastServerMsgTime > HEARTBEAT_STALE_MS) {
    console.warn('[WebSocket] stale after tab returned to foreground, forcing reconnect');
    try { socket.close(); } catch (e) { /* already closing */ }
  }
});

// Handle incoming messages
function onSocketMessage(event) {
  // Any traffic proves the connection is alive (see heartbeat)
  lastServerMsgTime = Date.now();

  const data = event.data;

  // Check if data is binary (Audio Chunk)
  if (data instanceof ArrayBuffer) {
    handleAudioChunk(data);
    return;
  }

  // Otherwise handle JSON messages
  try {
    const msg = JSON.parse(data);
    const eventName = msg.event;
    const payload = msg.data;

    switch (eventName) {
      case 'connect_ack':
        console.log('Connection confirmed:', payload);
        // 晚十一：连接即回带的生效设置（含持久化的前端默认值）恢复
        // UI 控件——刷新/换浏览器不丢设置。直接赋 value 不触发事件，
        // 不会被 debounce 回发。
        applyServerSettings(payload.settings);
        break;
      case 'pong':
        // heartbeat reply; liveness already recorded above
        break;
      case 'text_response':
        console.log('[LLM]', payload.text);
        appendAssistantText(payload.text);
        break;
      case 'turn_start':
        // Server starts a new reply pipeline: open a fresh assistant bubble
        startAssistantMessage();
        break;
      case 'config_applied':
        flashSavedFlag();
        console.log('[Config] applied:', payload);
        break;
      case 'vad_loading':
        handleVadLoading(payload);
        break;
      case 'vad_status':
        if (vadStatusEl) {
          vadStatusEl.textContent = payload.message || payload.state || "Unknown Status";
          vadStatusEl.dataset.state = payload.state || "idle";
        }
        break;
      case 'circle_status':
        updateCircleState(payload.status);
        break;
      case 'user_transcription':
        handleUserTranscription(payload);
        break;
      case 'stop_audio':
        console.log('Received stop_audio signal');
        if (ttsNode) ttsNode.port.postMessage({ type: 'abort' });
        updateCircleState("LISTENING");
        break;
      case 'pause_audio':
        if (ttsNode) ttsNode.port.postMessage({ type: 'pause' });
        updateCircleState("LISTENING");
        break;
      case 'resume_audio':
        if (ttsNode) ttsNode.port.postMessage({ type: 'resume' });
        updateCircleState('SPEAKING');
        break;
      case 'pvad_status':
        handlePvadStatus(payload);
        break;
      case 'pvad_reject':
        handlePvadReject(payload);
        break;
      case 'selftalk_drop':
        handleSelftalkDrop(payload);
        break;
      case 'task_update':
        handleTaskUpdate(payload);
        break;
      default:
        console.log('Unknown event:', eventName);
    }
  } catch (e) {
    console.error('Failed to parse message:', e);
  }
};


// DOM elements
const vadStatusEl = document.getElementById("vadStatus");

// Loading Overlay Elements
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMessage = document.getElementById("loadingMessage");
const loadingConfirmBtn = document.getElementById("loadingConfirmBtn");
const loadingSpinner = document.querySelector(".loading-spinner");
const loadingSuccess = document.querySelector(".loading-success");

function handleVadLoading({ state, message }) {
  // After the first confirmed session, reconnects must be transparent:
  // the server re-emits vad_loading for the new session, but making the
  // user click "Start Experience" again mid-conversation would look
  // exactly like the "page went dead" bug this fixes.
  if (loadingConfirmedOnce) {
    loadingOverlay.classList.add("hidden");
    return;
  }

  loadingOverlay.classList.remove("hidden");
  loadingMessage.textContent = message;

  if (state === "loading") {
    loadingConfirmBtn.classList.add("hidden");
    loadingSpinner.classList.remove("hidden");
    loadingSuccess.classList.add("hidden");
  } else if (state === "ready") {
    loadingSpinner.classList.add("hidden");
    loadingSuccess.classList.remove("hidden");
    loadingConfirmBtn.classList.remove("hidden");
  }
}

// Close loading overlay
loadingConfirmBtn.addEventListener("click", async () => {
  loadingConfirmedOnce = true;
  loadingOverlay.classList.add("hidden");
  updateCircleState("READY");

  // Initialize TTS Audio Engine
  if (!ttsContext) {
    await initTTSAudioEngine();
  }
});

// ==================== Chat log (right rail transcript) ====================
// The server sends `user_transcription` with the *growing* ASR buffer while
// the user speaks (VAD callback), plus the final text right before
// `turn_start` (pipeline, final: true). So: keep one live user bubble that
// is updated in place until its turn commits (frozen), then open a fresh
// bubble per later utterance, and a fresh assistant bubble on each
// `turn_start`.
const chatLog = document.getElementById('chatLog');
const MAX_LOG_MESSAGES = 200;
let lastMsg = null; // { role: 'user'|'assistant', body: HTMLElement }

function scrollLog() {
  // Stick to bottom unless the user scrolled up to read
  const nearBottom =
    chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 60;
  if (nearBottom) chatLog.scrollTop = chatLog.scrollHeight;
}

function addMessage(role, text) {
  const empty = chatLog.querySelector('.log-empty');
  if (empty) empty.remove();

  const msg = document.createElement('div');
  msg.className = 'msg msg-' + role;

  const roleEl = document.createElement('span');
  roleEl.className = 'msg-role';
  roleEl.textContent = role === 'user' ? 'YOU' : 'AI';

  const body = document.createElement('div');
  body.className = 'msg-body';
  body.textContent = text;

  msg.append(roleEl, body);
  chatLog.appendChild(msg);
  while (chatLog.children.length > MAX_LOG_MESSAGES) {
    chatLog.firstElementChild.remove();
  }
  scrollLog();
  return body;
}

function flashBody(body) {
  body.classList.add('updated');
  window.setTimeout(() => body.classList.remove('updated'), 400);
}

// A committed turn freezes its user bubble (2026-11-03): the server's
// final user_transcription (final: true) means the sentence is settled,
// so anything streamed after it (a phantom blip's partials — the 01:01
// "是。" swallow) opens its OWN bubble instead of overwriting the real
// sentence in place, and a selftalk_drop only ever removes that
// unfrozen phantom bubble, never the frozen one.
let userBubbleFinal = false;
// body of the last FROZEN user bubble — the fold target for a continued
// turn (append: true)
let lastFinalUserBody = null;
// Streaming-partial stability gate (2026-11-05): paraformer revises its
// live partials mid-utterance on near-homophones — "你数到哪了" briefly
// streamed as "你说" (数/说), "你再查" as "你是"/"你在这" — and every
// revision flashed as a phantom word in the live bubble (commits stay
// clean; this is display-only). A partial is therefore only shown once
// the NEXT partial confirms it as a stable prefix; a revision never
// reaches the screen, it just replaces the internal candidate. Costs one
// partial update (~160-320ms) of live-caption lag; the final text always
// renders verbatim. A lone partial (the "刚才" blip family) now shows
// nothing at all.
let stablePartial = null;   // prefix confirmed by a later partial
let pendingPartial = null;  // newest partial, not yet confirmed
// Unfrozen user bubbles opened by partials (2026-11-06 phantom 「你帮我」:
// an assistant turn_start landing BETWEEN two partials of one utterance
// re-split the live stream — the next partial opened a SECOND bubble, and
// the append-fold only removed the last one, orphaning the first with
// whatever the stability gate had confirmed ("你帮我") forever in the
// transcript. Every partial-opened bubble is tracked here and swept on
// final / fold / selftalk_drop, so a split can no longer leave one behind.)
let liveUserBodies = [];

function sweepLiveUserBodies(keep) {
  liveUserBodies = liveUserBodies.filter((b) => {
    if (b === keep) return false;       // kept element, just untracked
    if (!b.isConnected) return false;   // already removed elsewhere
    b.closest('.msg')?.remove();
    return false;
  });
}

function handleUserTranscription({ text, final, append }) {
  if (!text) return;
  if (!final && !append) {
    if (
      pendingPartial !== null &&
      text.length > pendingPartial.length &&
      text.startsWith(pendingPartial) &&
      (stablePartial === null || pendingPartial.length > stablePartial.length)
    ) {
      stablePartial = pendingPartial;
    }
    pendingPartial = text;
    if (stablePartial === null) return;  // nothing confirmed yet
    text = stablePartial;
  }
  if (append && final && lastFinalUserBody && lastFinalUserBody.isConnected) {
    // Continued utterance (server flagged it: previous user turn recent,
    // no assistant audio in between — the same merge the LLM applies via
    // groupby). Fold into the previous frozen bubble instead of opening a
    // third one, and drop the live partial bubble(s) this continuation was
    // streaming into — ALL of them: a turn_start between two partials
    // splits the stream into several bubbles, and only the newest is
    // lastMsg (2026-11-06 「你帮我」 phantom).
    sweepLiveUserBodies(null);
    lastFinalUserBody.textContent += text;
    lastMsg = { role: 'user', body: lastFinalUserBody };
    userBubbleFinal = true;
    stablePartial = pendingPartial = null;
    flashBody(lastFinalUserBody);
    scrollLog();
    return;
  }
  if (lastMsg && lastMsg.role === 'user' && !userBubbleFinal) {
    // Live update of the bubble being spoken into. The stability gate
    // often maps consecutive partials to the same display text — skip
    // the rewrite and the pulse when nothing changed.
    if (lastMsg.body.textContent !== text) {
      lastMsg.body.textContent = text;
      flashBody(lastMsg.body);
    } else if (!final) {
      return;
    }
  } else {
    lastMsg = { role: 'user', body: addMessage('user', text) };
    userBubbleFinal = false;
    if (!final) liveUserBodies.push(lastMsg.body);
  }
  if (final) {
    userBubbleFinal = true;
    lastFinalUserBody = lastMsg.body;
    stablePartial = pendingPartial = null;
    // Freeze keeps this bubble; any OTHER unfrozen partial bubbles (the
    // turn_start split) are stale stream fragments — remove them.
    sweepLiveUserBodies(lastMsg.body);
    flashBody(lastMsg.body);
  }
}

function startAssistantMessage() {
  lastMsg = { role: 'assistant', body: addMessage('assistant', '…') };
}

function appendAssistantText(text) {
  if (!text) return;
  if (!lastMsg || lastMsg.role !== 'assistant') {
    startAssistantMessage();
  }
  const body = lastMsg.body;
  if (body.textContent === '…') body.textContent = '';
  body.textContent += text;
  flashBody(body);
  scrollLog();
}

// ==================== TTS Engine (SharedArrayBuffer + AudioWorklet) ====================
async function initTTSAudioEngine() {
  try {
    // Create context with TTS specific sample rate (24k)
    ttsContext = new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' });

    // Add AudioWorklet Module
    await ttsContext.audioWorklet.addModule('processor.js');

    // Initialize SharedArrayBuffer
    // Header (8 bytes) + Data (Float32 * SIZE)
    const sabSize = 8 + TTS_BUFFER_SIZE * 4;
    ttsSab = new SharedArrayBuffer(sabSize);

    // Views
    ttsStates = new Int32Array(ttsSab, 0, 2); // [0]: WriteIndex, [1]: ReadIndex
    ttsFloat32Data = new Float32Array(ttsSab, 8);

    // Init Indices
    Atomics.store(ttsStates, 0, 0);
    Atomics.store(ttsStates, 1, 0);

    // Create Worklet Node
    ttsNode = new AudioWorkletNode(ttsContext, 'tts-processor', {
      processorOptions: { sab: ttsSab }
    });

    // Handle messages from Worklet (e.g., playback finished)
    ttsNode.port.onmessage = (e) => {
      if (e.data.type === 'playback_ended') {
        updateCircleState('LISTENING');
        console.log('[TTS] playback finished');
      }
    };

    ttsNode.connect(ttsContext.destination);
    console.log("TTS Audio Engine Initialized");

  } catch (e) {
    console.error("Failed to init TTS Engine:", e);
    alert("Audio Engine initialization failed! Only supported in localhost or https. Check console for details.");
  }
}

// Handle Receive Audio Chunk (Binary)
function handleAudioChunk(arrayBuffer) {
  if (!ttsSab || !ttsFloat32Data) return;

  // Resume context if suspended
  if (ttsContext.state === 'suspended') {
    ttsContext.resume();
  }

  // Convert incoming Int16 PCM to Float32
  // Assuming backend sends raw Int16 bytes from WAV
  const inputInt16 = new Int16Array(arrayBuffer);
  const inputLen = inputInt16.length;

  // Load current Write Index
  let writeIndex = Atomics.load(ttsStates, 0);

  for (let i = 0; i < inputLen; i++) {
    // Normalize Int16 to Float32 [-1, 1]
    const sample = inputInt16[i] / 32768.0;

    // Circular buffer write
    const bufferIndex = writeIndex % TTS_BUFFER_SIZE;
    ttsFloat32Data[bufferIndex] = sample;

    writeIndex++;
  }

  // Update Write Index atomically
  Atomics.store(ttsStates, 0, writeIndex);

  updateCircleState('SPEAKING');
}

// ==================== Audio Capture ====================
startBtn.addEventListener('click', async () => {
  try {
    listening = true;
    updateCircleState("LISTENING");
    startTimer();

    // Ensure TTS Engine is ready
    if (!ttsContext) await initTTSAudioEngine();
    if (ttsContext.state === 'suspended') ttsContext.resume();

    // Browser WebRTC processing chain (same as the FireRedChat demo):
    // AEC uses the tab's TTS playout as the echo reference, plus noise
    // suppression and AGC. This is the echo/denoise layer — the server
    // receives a cleaned stream and the pvad gate sees the same audio.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Separate context for recording to avoid sample rate mess
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

    // Send Audio Configuration
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        event: "config_audio",
        data: { sample_rate: audioContext.sampleRate }
      }));
    }

    // Ensure AudioContext is active (required by some browser policies)
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    dataArray = new Uint8Array(analyser.frequencyBinCount);

    // Create ScriptProcessor to send in frames
    processor = audioContext.createScriptProcessor(512, 1, 1);
    source.connect(analyser);
    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = e => {
      if (!listening) return;
      const input = e.inputBuffer.getChannelData(0);
      // Float32 -> Int16
      const int16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        int16[i] = Math.max(-1, Math.min(1, input[i])) * 0x7fff;
      }

      // Native WebSocket Send (Binary)
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(int16.buffer);
      }
    };

    startBtn.disabled = true;
    stopBtn.disabled = false;
    fitCanvas();
    drawUserWaveform();
    pulseAssistantCircle();

  } catch (err) {
    console.error('Cannot access microphone:', err);
    alert("Cannot access microphone! Please check browser permissions.");
    updateCircleState("READY");
    stopTimer();
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener('click', () => {
  listening = false;
  stopTimer();

  // Send Stop signal via JSON
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ event: 'duplex_stop' }));
  }

  // Interrupt TTS
  if (ttsNode) ttsNode.port.postMessage({ type: 'abort' });

  cancelAnimationFrame(animationId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  voiceCircle.style.transform = 'scale(1)';
  ctx.fillStyle = '#080b10';
  ctx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  updateCircleState("READY");
});

// ==================== Draw Waveform (neon on dark) ====================
function drawUserWaveform() {
  if (!listening) return;
  animationId = requestAnimationFrame(drawUserWaveform);

  const w = waveformCanvas.width;
  const h = waveformCanvas.height;

  analyser.getByteTimeDomainData(dataArray);
  ctx.fillStyle = '#080b10';
  ctx.fillRect(0, 0, w, h);

  // Center hairline
  ctx.strokeStyle = 'rgba(34, 211, 238, 0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Neon trace
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#22d3ee';
  ctx.shadowColor = 'rgba(34, 211, 238, 0.75)';
  ctx.shadowBlur = 8;
  ctx.beginPath();

  const sliceWidth = w / dataArray.length;
  let x = 0;
  let lastY = h / 2;

  for (let i = 0; i < dataArray.length; i++) {
    const v = dataArray[i] / 128.0;
    const y = lastY + (v * h / 2 - lastY) * 0.2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    lastY = y;
    x += sliceWidth;
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// ==================== Assistant Circle Animation ====================
let pulseDirection = 1;
function pulseAssistantCircle() {
  if (!listening) return;
  animationId = requestAnimationFrame(pulseAssistantCircle);

  // Only animate scale via JS if NOT in CSS-animated states (like speaking/processing)
  if (voiceCircle.classList.contains('state-speaking') || voiceCircle.classList.contains('state-processing')) {
    return;
  }

  let currentScale = parseFloat(voiceCircle.style.transform.replace('scale(', '').replace(')', '')) || 1;
  if (currentScale >= 1.05) pulseDirection = -1;
  if (currentScale <= 1) pulseDirection = 1;
  currentScale += pulseDirection * 0.002;
  voiceCircle.style.transform = `scale(${currentScale})`;
}

// ==================== Update Circle State ====================
function updateCircleState(state) {
  // Reset classes (orb + sonar-ring wrapper stay in sync)
  voiceCircle.classList.remove('state-speaking', 'state-listening', 'state-processing');
  orbWrap.classList.remove('listening', 'processing', 'speaking');

  let statusText = "READY";

  switch (state) {
    case 'LISTENING':
      voiceCircle.classList.add('state-listening');
      orbWrap.classList.add('listening');
      statusText = "LISTENING";
      break;

    case 'THINKING':
      voiceCircle.classList.add('state-processing');
      orbWrap.classList.add('processing');
      statusText = "THINKING";
      break;

    case 'SPEAKING':
      voiceCircle.classList.add('state-speaking');
      orbWrap.classList.add('speaking');
      statusText = "SPEAKING";
      break;

    case 'READY':
    default:
      statusText = "READY";
      break;
  }

  if (circleStatusEl) circleStatusEl.textContent = statusText;
}

// ==================== Live Settings Panel ====================
// Mirrors backend DEFAULT_SETTINGS (app.py). Sent via `update_config`;
// the server merges per-session and echoes back `config_applied`.
const DEFAULT_PERSONA =
  "你是 Elva，一个温柔、自然的语音对话助手，正在通过语音和用户实时聊天。" +
  "请用简短、口语化的自然中文回答，就像真人说话一样；不要输出多余解释，不要用列表或 Markdown 格式。" +
  "用户表示同意、附和或只是简单回应时，顺着上文继续说即可。" +
  "说话要像真人：应承、附和、告别这些高频话不要每次都用一模一样的句子，" +
  "开口前看一眼自己刚才说过什么，刚用过的句式就换一个说法——重复的固定句式听起来非常机械。" +
  "用户让你停下时，就什么都不要说。" +
  "身份必须始终如一：你的名字是 Elva。被问到“你是谁”时，用一两句口语化的中文介绍自己是 Elva。" +
  "你不是小爱同学、不是 Siri、不是小度，也不是任何手机或其他公司自带的语音助手——" +
  "永远不要用这些身份自称或介绍自己。";

const PERSONA_PRESETS = {
  default: DEFAULT_PERSONA,
  concise:
    "You are a voice assistant who answers as briefly as possible — one or two short sentences at most. " +
    "Speak naturally and colloquially, never use lists or markdown. " +
    "If the user agrees or backchannels, continue with the key point only.",
  storyteller:
    "You are a warm storyteller talking with the user over voice. " +
    "Tell vivid, immersive little stories or anecdotes, 3-5 sentences per turn, ending with a small hook that invites the user to respond. " +
    "Keep language spoken-style and simple; no lists or markdown.",
  tutor:
    "You are a patient tutor explaining things over voice. " +
    "Break every explanation into small spoken steps, use simple everyday analogies, and check understanding with one short question at the end of each turn. " +
    "Never use lists or markdown; keep each turn under five sentences.",
};

const settingsEls = {
  persona: document.getElementById('personaPrompt'),
  preset: document.getElementById('personaPreset'),
  applyBtn: document.getElementById('applyPersonaBtn'),
  resetBtn: document.getElementById('resetPersonaBtn'),
  saved: document.getElementById('personaSaved'),
  ttsEngine: document.getElementById('ttsEngine'),
  turnEngine: document.getElementById('turnEngine'),
  turnWaitMs: document.getElementById('turnWaitMs'),
  turnWaitMsVal: document.getElementById('turnWaitMsVal'),
  turnGraceMs: document.getElementById('turnGraceMs'),
  turnGraceMsVal: document.getElementById('turnGraceMsVal'),
  asrCorrect: document.getElementById('asrCorrect'),
  asrCorrectWaitMs: document.getElementById('asrCorrectWaitMs'),
  asrCorrectWaitMsVal: document.getElementById('asrCorrectWaitMsVal'),
  ttsCfg: document.getElementById('ttsCfg'),
  ttsCfgVal: document.getElementById('ttsCfgVal'),
  ttsSteps: document.getElementById('ttsSteps'),
  ttsStepsVal: document.getElementById('ttsStepsVal'),
  maxHistory: document.getElementById('maxHistory'),
  maxHistoryVal: document.getElementById('maxHistoryVal'),
  minSegLen: document.getElementById('minSegLen'),
  minSegLenVal: document.getElementById('minSegLenVal'),
  pvadMode: document.getElementById('pvadMode'),
  pvadThreshold: document.getElementById('pvadThreshold'),
  pvadThresholdVal: document.getElementById('pvadThresholdVal'),
  pvadStreamGate: document.getElementById('pvadStreamGate'),
  speakerSep: document.getElementById('speakerSep'),
  pvadEnrollBtn: document.getElementById('pvadEnrollBtn'),
  pvadEnrolled: document.getElementById('pvadEnrolled'),
  pvadProb: document.getElementById('pvadProb'),
  pvadNote: document.getElementById('pvadNote'),
  agentEnabled: document.getElementById('agentEnabled'),
  agentRuntimeMode: document.getElementById('agentRuntimeMode'),
  playbackGuard: document.getElementById('playbackGuard'),
  deliveryBargeinChunks: document.getElementById('deliveryBargeinChunks'),
  deliveryBargeinChunksVal: document.getElementById('deliveryBargeinChunksVal'),
  deliveryBargeinSpanMs: document.getElementById('deliveryBargeinSpanMs'),
  deliveryBargeinSpanMsVal: document.getElementById('deliveryBargeinSpanMsVal'),
  taskList: document.getElementById('taskList'),
  taskEmpty: document.getElementById('taskEmpty'),
  taskCount: document.getElementById('taskCount'),
};

function currentSettings() {
  return {
    system_prompt: settingsEls.persona.value.trim() || DEFAULT_PERSONA,
    max_history: parseInt(settingsEls.maxHistory.value, 10) || 5,
    min_seg_len: parseInt(settingsEls.minSegLen.value, 10) || 10,
    tts_engine: settingsEls.ttsEngine.value,
    turn_engine: settingsEls.turnEngine.value,
    turn_wait_ms: parseInt(settingsEls.turnWaitMs.value, 10) || 800,
    turn_grace_ms: parseInt(settingsEls.turnGraceMs.value, 10) || 900,
    tts_cfg: parseFloat(settingsEls.ttsCfg.value) || 2.0,
    tts_timesteps: parseInt(settingsEls.ttsSteps.value, 10) || 10,
    pvad_mode: settingsEls.pvadMode.value,
    pvad_threshold: parseFloat(settingsEls.pvadThreshold.value) || 0.5,
    pvad_stream_gate: settingsEls.pvadStreamGate.checked,
    speaker_sep: settingsEls.speakerSep.checked,
    agent_enabled: settingsEls.agentEnabled.checked,
    agent_runtime_mode: settingsEls.agentRuntimeMode.value,
    playback_guard: settingsEls.playbackGuard.checked,
    delivery_bargein_chunks: parseInt(settingsEls.deliveryBargeinChunks.value, 10) || 5,
    delivery_bargein_span_ms: parseInt(settingsEls.deliveryBargeinSpanMs.value, 10) || 1000,
    asr_correct: settingsEls.asrCorrect.checked,
    asr_correct_wait_ms: parseInt(settingsEls.asrCorrectWaitMs.value, 10) || 500,
  };
}

function sendConfig(partial) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    // persist: true —— 晚十一：真实 UI 的设置跨会话持久化为后续默认值
    // （测试脚本/未知客户端不带此标志，不写默认值文件）
    socket.send(JSON.stringify({ event: 'update_config', data: partial, persist: true }));
  }
}

// --- server settings → UI (connect_ack carries the session's effective
// settings, including the persisted frontend defaults) ---
function applyServerSettings(s) {
  if (!s || typeof s !== 'object') return;
  const set = (el, v) => { if (el && v !== undefined && v !== null) el.value = v; };
  if (s.system_prompt) settingsEls.persona.value = s.system_prompt;
  set(settingsEls.ttsEngine, s.tts_engine);
  set(settingsEls.turnEngine, s.turn_engine);
  set(settingsEls.turnWaitMs, s.turn_wait_ms);
  settingsEls.turnWaitMsVal.textContent = settingsEls.turnWaitMs.value + ' ms';
  set(settingsEls.turnGraceMs, s.turn_grace_ms);
  settingsEls.turnGraceMsVal.textContent = settingsEls.turnGraceMs.value + ' ms';
  set(settingsEls.deliveryBargeinChunks, s.delivery_bargein_chunks);
  settingsEls.deliveryBargeinChunksVal.textContent = settingsEls.deliveryBargeinChunks.value;
  set(settingsEls.deliveryBargeinSpanMs, s.delivery_bargein_span_ms);
  settingsEls.deliveryBargeinSpanMsVal.textContent = settingsEls.deliveryBargeinSpanMs.value + ' ms';
  set(settingsEls.asrCorrectWaitMs, s.asr_correct_wait_ms);
  settingsEls.asrCorrectWaitMsVal.textContent = settingsEls.asrCorrectWaitMs.value + ' ms';
  set(settingsEls.ttsCfg, s.tts_cfg);
  settingsEls.ttsCfgVal.textContent = parseFloat(settingsEls.ttsCfg.value).toFixed(1);
  set(settingsEls.ttsSteps, s.tts_timesteps);
  settingsEls.ttsStepsVal.textContent = settingsEls.ttsSteps.value;
  set(settingsEls.maxHistory, s.max_history);
  settingsEls.maxHistoryVal.textContent = settingsEls.maxHistory.value;
  set(settingsEls.minSegLen, s.min_seg_len);
  settingsEls.minSegLenVal.textContent = settingsEls.minSegLen.value;
  set(settingsEls.pvadMode, s.pvad_mode);
  set(settingsEls.pvadThreshold, s.pvad_threshold);
  settingsEls.pvadThresholdVal.textContent = parseFloat(settingsEls.pvadThreshold.value).toFixed(2);
  set(settingsEls.agentRuntimeMode, s.agent_runtime_mode);
  if (s.playback_guard !== undefined) settingsEls.playbackGuard.checked = !!s.playback_guard;
  if (s.asr_correct !== undefined) settingsEls.asrCorrect.checked = !!s.asr_correct;
  if (s.pvad_stream_gate !== undefined) settingsEls.pvadStreamGate.checked = !!s.pvad_stream_gate;
  if (s.speaker_sep !== undefined) settingsEls.speakerSep.checked = !!s.speaker_sep;
  if (s.agent_enabled !== undefined) settingsEls.agentEnabled.checked = !!s.agent_enabled;
  syncTurnTimingState();
}

function sendCurrentSettings() {
  sendConfig(currentSettings());
}

function flashSavedFlag() {
  if (!settingsEls.saved) return;
  settingsEls.saved.classList.remove('hidden');
  window.setTimeout(() => settingsEls.saved.classList.add('hidden'), 1500);
}

// --- persona textarea / presets ---
settingsEls.persona.value = DEFAULT_PERSONA;
settingsEls.preset.addEventListener('change', () => {
  const key = settingsEls.preset.value;
  if (key && PERSONA_PRESETS[key]) {
    settingsEls.persona.value = PERSONA_PRESETS[key];
    sendConfig({ system_prompt: PERSONA_PRESETS[key] });
  }
});
settingsEls.applyBtn.addEventListener('click', () => {
  sendConfig({ system_prompt: settingsEls.persona.value.trim() || DEFAULT_PERSONA });
});
settingsEls.resetBtn.addEventListener('click', () => {
  settingsEls.persona.value = DEFAULT_PERSONA;
  settingsEls.preset.value = 'default';
  sendConfig({ system_prompt: DEFAULT_PERSONA });
});

// --- debounced live sliders / selects ---
let configDebounce = null;
function debouncedSend() {
  if (configDebounce) clearTimeout(configDebounce);
  configDebounce = setTimeout(() => {
    const s = currentSettings();
    sendConfig({
      tts_engine: s.tts_engine,
      turn_engine: s.turn_engine,
      turn_wait_ms: s.turn_wait_ms,
      turn_grace_ms: s.turn_grace_ms,
      tts_cfg: s.tts_cfg,
      tts_timesteps: s.tts_timesteps,
      max_history: s.max_history,
      min_seg_len: s.min_seg_len,
      pvad_mode: s.pvad_mode,
      pvad_threshold: s.pvad_threshold,
      pvad_stream_gate: s.pvad_stream_gate,
      speaker_sep: s.speaker_sep,
      agent_enabled: s.agent_enabled,
      agent_runtime_mode: s.agent_runtime_mode,
      playback_guard: s.playback_guard,
      delivery_bargein_chunks: s.delivery_bargein_chunks,
      delivery_bargein_span_ms: s.delivery_bargein_span_ms,
      asr_correct: s.asr_correct,
      asr_correct_wait_ms: s.asr_correct_wait_ms,
    });
  }, 400);
}

// The two turn-timing sliders only affect the SoulX engine — gray them out
// while frc is selected (values still ride along in the payload, the server
// ignores them in frc mode).
function syncTurnTimingState() {
  const frc = settingsEls.turnEngine.value === 'frc';
  [settingsEls.turnWaitMs, settingsEls.turnGraceMs, settingsEls.playbackGuard,
   settingsEls.deliveryBargeinChunks, settingsEls.deliveryBargeinSpanMs,
   settingsEls.asrCorrect, settingsEls.asrCorrectWaitMs].forEach((el) => {
    el.disabled = frc;
    el.closest('.field').classList.toggle('frc-off', frc);
  });
}

settingsEls.ttsEngine.addEventListener('change', debouncedSend);
settingsEls.turnEngine.addEventListener('change', () => {
  syncTurnTimingState();
  debouncedSend();
});
settingsEls.turnWaitMs.addEventListener('input', () => {
  settingsEls.turnWaitMsVal.textContent = settingsEls.turnWaitMs.value + ' ms';
  debouncedSend();
});
settingsEls.turnGraceMs.addEventListener('input', () => {
  settingsEls.turnGraceMsVal.textContent = settingsEls.turnGraceMs.value + ' ms';
  debouncedSend();
});
settingsEls.asrCorrect.addEventListener('change', debouncedSend);
settingsEls.deliveryBargeinChunks.addEventListener('input', () => {
  settingsEls.deliveryBargeinChunksVal.textContent = settingsEls.deliveryBargeinChunks.value;
  debouncedSend();
});
settingsEls.deliveryBargeinSpanMs.addEventListener('input', () => {
  settingsEls.deliveryBargeinSpanMsVal.textContent = settingsEls.deliveryBargeinSpanMs.value + ' ms';
  debouncedSend();
});
settingsEls.asrCorrectWaitMs.addEventListener('input', () => {
  settingsEls.asrCorrectWaitMsVal.textContent = settingsEls.asrCorrectWaitMs.value + ' ms';
  debouncedSend();
});
syncTurnTimingState();
settingsEls.ttsCfg.addEventListener('input', () => {
  settingsEls.ttsCfgVal.textContent = parseFloat(settingsEls.ttsCfg.value).toFixed(1);
  debouncedSend();
});
settingsEls.ttsSteps.addEventListener('input', () => {
  settingsEls.ttsStepsVal.textContent = settingsEls.ttsSteps.value;
  debouncedSend();
});
settingsEls.maxHistory.addEventListener('input', () => {
  settingsEls.maxHistoryVal.textContent = settingsEls.maxHistory.value;
  debouncedSend();
});
settingsEls.minSegLen.addEventListener('input', () => {
  settingsEls.minSegLenVal.textContent = settingsEls.minSegLen.value;
  debouncedSend();
});

// ==================== Speaker Gate (pvad sidecar) ====================
settingsEls.pvadMode.addEventListener('change', debouncedSend);
settingsEls.agentRuntimeMode.addEventListener('change', debouncedSend);
settingsEls.pvadStreamGate.addEventListener('change', debouncedSend);
settingsEls.speakerSep.addEventListener('change', debouncedSend);
settingsEls.pvadThreshold.addEventListener('input', () => {
  settingsEls.pvadThresholdVal.textContent =
    parseFloat(settingsEls.pvadThreshold.value).toFixed(2);
  debouncedSend();
});
settingsEls.pvadEnrollBtn.addEventListener('click', () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ event: 'pvad_enroll' }));
    flashGateNote('Re-enrollment requested…');
  }
});

let gateNoteTimer = null;

function flashGateNote(text, sticky = false) {
  if (!settingsEls.pvadNote) return;
  settingsEls.pvadNote.textContent = text;
  settingsEls.pvadNote.classList.remove('hidden');
  if (gateNoteTimer) clearTimeout(gateNoteTimer);
  if (!sticky) {
    gateNoteTimer = setTimeout(() => settingsEls.pvadNote.classList.add('hidden'), 3000);
  }
}

function handlePvadStatus({ enrolled, prob, mode, threshold, gated }) {
  if (settingsEls.pvadEnrolled) {
    settingsEls.pvadEnrolled.dataset.state = enrolled ? 'enrolled' : 'off';
    settingsEls.pvadEnrolled.textContent = enrolled
      ? (gated ? 'enrolled · muting others' : 'enrolled')
      : 'not enrolled';
  }
  if (settingsEls.pvadProb) {
    settingsEls.pvadProb.textContent = prob == null ? '–' : prob.toFixed(2);
  }
}

function handlePvadReject({ kind, cos, threshold, shadow }) {
  const what = kind === 'bargein' ? 'Barge-in' : 'Turn';
  flashGateNote(
    `${what} ignored — not the main speaker (similarity ${cos.toFixed(2)} < ${threshold})` +
    (shadow ? ' [shadow]' : ''),
    !shadow
  );
  console.log('[pvad] rejected', kind, { cos, threshold, shadow });
}

// Echo of our own TTS / a breath blip completed a phantom turn
// server-side and was dropped (in-flight self-talk check, 2026-11-02):
// remove its stale live bubble — the few characters the mic picked up —
// so they don't linger in the chat log as a user message that was never
// answered. Never touches a FROZEN bubble (a real committed sentence):
// the phantom's partials opened their own unfrozen bubble after the
// freeze, so only that one is stale.
function handleSelftalkDrop({ reason } = {}) {
  if (lastMsg && lastMsg.role === 'user' && !userBubbleFinal) {
    lastMsg.body.parentElement?.remove();
    lastMsg = null;
  }
  // the dropped turn's partial candidates are stale too — and so are any
  // other unfrozen partial bubbles a mid-stream turn_start split off
  sweepLiveUserBodies(null);
  stablePartial = pendingPartial = null;
  flashGateNote(reason === 'blip'
    ? '已忽略幻影语音片段（未检测到有效人声）'
    : '已忽略回声（识别到的是我自己的声音）');
}

// ==================== Background Tasks (fast/slow brain agent) ====================
// task_update payloads: {task_id, goal, status, progress, result}
// status: queued | running | completed | delivered | interrupted | failed | canceled
const TASK_STATUS_LABEL = {
  queued: 'queued', running: 'running', completed: 'done',
  delivered: 'reported', interrupted: 'cut short', failed: 'failed',
  canceled: 'canceled',
};
const MAX_TASK_CARDS = 10;

settingsEls.agentEnabled.addEventListener('change', () => {
  // immediate (not debounced): switching the agent off cancels running
  // background tasks server-side
  sendConfig({ agent_enabled: settingsEls.agentEnabled.checked });
});

function handleTaskUpdate({ task_id, goal, status, progress, result }) {
  if (!settingsEls.taskList) return;
  let card = settingsEls.taskList.querySelector(`[data-task="${task_id}"]`);
  if (!card) {
    card = document.createElement('div');
    card.className = 'task-card';
    card.dataset.task = task_id;

    const head = document.createElement('div');
    head.className = 'task-head';
    const id = document.createElement('span');
    id.className = 'task-id';
    id.textContent = task_id;
    const st = document.createElement('span');
    st.className = 'task-status';
    head.append(id, st);

    const g = document.createElement('p');
    g.className = 'task-goal';

    const pr = document.createElement('p');
    pr.className = 'task-progress';

    const res = document.createElement('p');
    res.className = 'task-result';

    card.append(head, g, pr, res);
    settingsEls.taskList.prepend(card);
    // oldest cards fall off the bottom
    while (settingsEls.taskList.children.length > MAX_TASK_CARDS) {
      settingsEls.taskList.lastElementChild.remove();
    }
  }

  card.dataset.status = status;
  card.querySelector('.task-status').dataset.s = status;
  card.querySelector('.task-status').textContent =
    TASK_STATUS_LABEL[status] || status;
  card.querySelector('.task-goal').textContent = goal || '';
  const pr = card.querySelector('.task-progress');
  pr.textContent = progress || '';
  pr.style.display = progress ? '' : 'none';
  const res = card.querySelector('.task-result');
  res.textContent = result || '';
  res.style.display = result ? '' : 'none';

  const n = settingsEls.taskList.children.length;
  if (settingsEls.taskCount) {
    settingsEls.taskCount.textContent = String(n);
    settingsEls.taskCount.dataset.state = n ? 'enrolled' : 'off';
  }
  if (settingsEls.taskEmpty) {
    settingsEls.taskEmpty.style.display = n ? 'none' : '';
  }
}
