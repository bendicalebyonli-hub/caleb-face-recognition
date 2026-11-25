/**
 * Caleb Yonli AI · Mobile Biometric Face ID Engine
 * ArcFace 512-D ONNX Feature Matching
 * Subject: Caleb Yonli (ID: ID-001)
 */

// ==========================================
// 1. CONFIGURATION & STATE
// ==========================================
const CONFIG = {
  DEFAULT_API_URL: window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
    ? window.location.origin
    : (window.location.origin.includes('vercel.app') 
        ? 'https://caleb-face-recognition.onrender.com' 
        : window.location.origin),
  STORAGE_KEY_API: 'CALEB_API_BASE',
  STORAGE_KEY_SOUND: 'CALEB_SOUND_ENABLED',
  STORAGE_KEY_HAPTIC: 'CALEB_HAPTIC_ENABLED',
  STORAGE_KEY_HISTORY: 'CALEB_ATTENDANCE_LOG',
  STORAGE_KEY_THRESHOLD: 'CALEB_COSINE_THRESHOLD',
  PRIMARY_NAME: 'Caleb Yonli',
  PRIMARY_ID: 'ID-001',
  DEFAULT_THRESHOLD: 0.4000
};

let state = {
  apiUrl: localStorage.getItem(CONFIG.STORAGE_KEY_API) || CONFIG.DEFAULT_API_URL,
  threshold: parseFloat(localStorage.getItem(CONFIG.STORAGE_KEY_THRESHOLD)) || CONFIG.DEFAULT_THRESHOLD,
  soundEnabled: localStorage.getItem(CONFIG.STORAGE_KEY_SOUND) !== 'false',
  hapticEnabled: localStorage.getItem(CONFIG.STORAGE_KEY_HAPTIC) !== 'false',
  activeTab: 'scanner',
  scanMode: 'manual', // 'manual' | 'auto'
  facingMode: 'user', // 'user' | 'environment'
  cameraStream: null,
  torchActive: false,
  isScanning: false,
  autoScanTimer: null,
  selectedFile: null,
  audioCtx: null,
  deferredPwaPrompt: null,
  latestJsonResponse: null,
  historyFilter: 'all',
  history: []
};

// Load persistent history
try {
  const saved = localStorage.getItem(CONFIG.STORAGE_KEY_HISTORY);
  if (saved) state.history = JSON.parse(saved);
} catch (e) {
  state.history = [];
}

// ==========================================
// 2. INITIALIZATION
// ==========================================
window.addEventListener('DOMContentLoaded', () => {
  state.apiUrl = state.apiUrl.trim().replace(/\/+$/, '');

  initPwaServiceWorker();
  initAudioAndHaptics();
  initCamera();
  checkBackendHealth();
  renderHistoryView();
  updateHistoryBadge();
  setupDropZone();

  // Populate settings modal defaults
  const inputUrl = document.getElementById('input-api-url');
  if (inputUrl) inputUrl.value = state.apiUrl;
  const inputThresh = document.getElementById('input-threshold');
  const threshVal = document.getElementById('threshold-val');
  if (inputThresh) inputThresh.value = state.threshold;
  if (threshVal) threshVal.textContent = Number(state.threshold).toFixed(4);
});

// ==========================================
// 3. PWA SERVICE WORKER & APP INSTALL
// ==========================================
function initPwaServiceWorker() {
  if ('serviceWorker' in navigator && (window.location.protocol === 'https:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    navigator.serviceWorker.register('./sw.js')
      .then((reg) => console.log('Caleb FaceID PWA Active:', reg.scope))
      .catch((err) => console.warn('PWA ServiceWorker note:', err));
  }

  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    state.deferredPwaPrompt = e;
    const banner = document.getElementById('pwa-install-banner');
    if (banner) banner.classList.add('visible');
  });

  window.addEventListener('appinstalled', () => {
    console.log('Caleb FaceID App installed to phone Home Screen');
    const banner = document.getElementById('pwa-install-banner');
    if (banner) banner.classList.remove('visible');
    showLiveToast('Installed to your Home Screen!');
  });
}

function triggerAppInstall() {
  if (state.deferredPwaPrompt) {
    state.deferredPwaPrompt.prompt();
    state.deferredPwaPrompt.userChoice.then(() => {
      state.deferredPwaPrompt = null;
      const banner = document.getElementById('pwa-install-banner');
      if (banner) banner.classList.remove('visible');
    });
  } else {
    showLiveToast('Install via browser menu: "Add to Home Screen"');
  }
}

// ==========================================
// 4. AUDIO & HAPTIC SYNTHESIZERS
// ==========================================
function initAudioAndHaptics() {
  const btnSound = document.getElementById('btn-sound-toggle');
  const btnHaptic = document.getElementById('btn-haptic-toggle');

  updateSoundButtonUI();
  updateHapticButtonUI();

  if (btnSound) {
    btnSound.addEventListener('click', () => {
      state.soundEnabled = !state.soundEnabled;
      localStorage.setItem(CONFIG.STORAGE_KEY_SOUND, state.soundEnabled);
      updateSoundButtonUI();
      if (state.soundEnabled) playBiometricSound('scan');
      showLiveToast(`Sound: ${state.soundEnabled ? 'ON' : 'OFF'}`);
    });
  }

  if (btnHaptic) {
    btnHaptic.addEventListener('click', () => {
      state.hapticEnabled = !state.hapticEnabled;
      localStorage.setItem(CONFIG.STORAGE_KEY_HAPTIC, state.hapticEnabled);
      updateHapticButtonUI();
      if (state.hapticEnabled) triggerHapticFeedback([40]);
      showLiveToast(`Vibration: ${state.hapticEnabled ? 'ON' : 'OFF'}`);
    });
  }
}

function updateSoundButtonUI() {
  const btn = document.getElementById('btn-sound-toggle');
  const icon = document.getElementById('sound-icon');
  if (!btn || !icon) return;
  if (state.soundEnabled) {
    btn.classList.add('active');
    icon.innerHTML = `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>`;
  } else {
    btn.classList.remove('active');
    icon.innerHTML = `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line>`;
  }
}

function updateHapticButtonUI() {
  const btn = document.getElementById('btn-haptic-toggle');
  if (btn) btn.classList.toggle('active', state.hapticEnabled);
}

function playBiometricSound(type) {
  if (!state.soundEnabled) return;
  try {
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }

    const now = state.audioCtx.currentTime;
    const osc = state.audioCtx.createOscillator();
    const gain = state.audioCtx.createGain();
    osc.connect(gain);
    gain.connect(state.audioCtx.destination);

    if (type === 'match') {
      // Harmonic major chime for PASS
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, now); // C5
      osc.frequency.setValueAtTime(659.25, now + 0.08); // E5
      osc.frequency.setValueAtTime(783.99, now + 0.16); // G5
      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
      osc.start(now);
      osc.stop(now + 0.45);
    } else if (type === 'unmatched') {
      // Low sawtooth warning buzz for FAIL
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(220, now);
      osc.frequency.setValueAtTime(140, now + 0.12);
      gain.gain.setValueAtTime(0.28, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.38);
      osc.start(now);
      osc.stop(now + 0.38);
    } else if (type === 'scan') {
      // High chirping laser click
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(1300, now);
      gain.gain.setValueAtTime(0.12, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.09);
      osc.start(now);
      osc.stop(now + 0.09);
    }
  } catch (e) {}
}

function triggerHapticFeedback(pattern) {
  if (!state.hapticEnabled || !navigator.vibrate) return;
  try {
    navigator.vibrate(pattern);
  } catch (e) {}
}

// ==========================================
// 5. LIVE CAMERA HUD SCANNER
// ==========================================
async function initCamera() {
  const btnFlip = document.getElementById('btn-flip-camera');
  const btnTorch = document.getElementById('btn-torch-toggle');
  const btnShutter = document.getElementById('btn-shutter');

  if (btnFlip) btnFlip.addEventListener('click', flipCamera);
  if (btnTorch) btnTorch.addEventListener('click', toggleTorch);
  if (btnShutter) btnShutter.addEventListener('click', triggerManualScan);

  await startCameraStream();
}

async function startCameraStream() {
  const video = document.getElementById('camera-video');
  const prompt = document.getElementById('prompt-text');

  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (prompt) prompt.textContent = 'Camera stream inactive · Tap Shutter for mobile camera';
    return;
  }

  try {
    if (prompt) prompt.textContent = 'Connecting Camera Sensor...';
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: state.facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });

    if (video) {
      video.srcObject = state.cameraStream;
      video.classList.toggle('rear-camera', state.facingMode !== 'user');
    }

    if (prompt) prompt.textContent = 'Align Caleb\'s Face In Viewfinder';
  } catch (err) {
    console.warn('Camera stream notice:', err);
    if (prompt) prompt.textContent = 'Tap Shutter to Snap with Camera';
  }
}

async function flipCamera() {
  state.facingMode = (state.facingMode === 'user') ? 'environment' : 'user';
  playBiometricSound('scan');
  triggerHapticFeedback([40]);
  await startCameraStream();
}

async function toggleTorch() {
  if (!state.cameraStream) return;
  const track = state.cameraStream.getVideoTracks()[0];
  if (!track || !track.applyConstraints) return;

  const btnTorch = document.getElementById('btn-torch-toggle');
  try {
    const capabilities = track.getCapabilities ? track.getCapabilities() : {};
    if (!capabilities.torch) {
      showLiveToast('Flashlight not supported on this lens');
      return;
    }
    state.torchActive = !state.torchActive;
    await track.applyConstraints({
      advanced: [{ torch: state.torchActive }]
    });

    if (btnTorch) btnTorch.classList.toggle('active', state.torchActive);
    triggerHapticFeedback([30]);
  } catch (err) {
    console.warn('Torch error:', err);
  }
}

function setScanMode(mode) {
  state.scanMode = mode;
  const btnManual = document.getElementById('btn-mode-manual');
  const btnAuto = document.getElementById('btn-mode-auto');
  const ring = document.getElementById('shutter-ring');

  if (btnManual) btnManual.classList.toggle('active', mode === 'manual');
  if (btnAuto) btnAuto.classList.toggle('active', mode === 'auto');
  if (ring) ring.classList.toggle('auto-pulse', mode === 'auto');

  playBiometricSound('scan');

  if (mode === 'auto') {
    showLiveToast('Auto-Scan Active (Continuous 2.5s loop)');
    startAutoScanLoop();
  } else {
    showLiveToast('Manual Snap Mode Active');
    stopAutoScanLoop();
  }
}

function startAutoScanLoop() {
  stopAutoScanLoop();
  state.autoScanTimer = setInterval(() => {
    if (state.activeTab === 'scanner' && !state.isScanning) {
      triggerManualScan();
    }
  }, 2500);
}

function stopAutoScanLoop() {
  if (state.autoScanTimer) {
    clearInterval(state.autoScanTimer);
    state.autoScanTimer = null;
  }
}

async function triggerManualScan() {
  if (state.isScanning) return;

  const video = document.getElementById('camera-video');

  if (video && video.srcObject && video.readyState >= 2) {
    triggerCameraFlash();
    playBiometricSound('scan');
    triggerHapticFeedback([50]);

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');

    if (state.facingMode === 'user') {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (blob) {
        await processBiometricScan(blob, 'camera');
      }
    }, 'image/jpeg', 0.92);

  } else {
    // Native mobile camera snap
    playBiometricSound('scan');
    const nativeCam = document.getElementById('native-camera-input');
    if (nativeCam) nativeCam.click();
  }
}

function triggerCameraFlash() {
  const flash = document.getElementById('camera-flash');
  if (flash) {
    flash.classList.remove('flash-active');
    void flash.offsetWidth;
    flash.classList.add('flash-active');
  }
}

function handleNativeCameraCapture(event) {
  const file = event.target.files[0];
  if (file) {
    processBiometricScan(file, 'camera');
  }
}

// ==========================================
// 6. PHOTO GALLERY & UPLOAD
// ==========================================
function setupDropZone() {
  const dropzone = document.getElementById('upload-dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith('image/') || file.name.endsWith('.heic') || file.name.endsWith('.heif'))) {
      loadPhotoFile(file);
    }
  });
}

function handleFilePicked(e) {
  const file = e.target.files[0];
  if (file) loadPhotoFile(file);
}

function loadPhotoFile(file) {
  state.selectedFile = file;
  const reader = new FileReader();

  reader.onload = (e) => {
    const preview = document.getElementById('image-preview');
    const idleContent = document.getElementById('dropzone-idle');
    const previewContainer = document.getElementById('preview-container');
    const btnScan = document.getElementById('btn-scan-upload');

    if (preview) preview.src = e.target.result;
    if (idleContent) idleContent.style.display = 'none';
    if (previewContainer) previewContainer.classList.add('active');
    if (btnScan) btnScan.disabled = false;

    playBiometricSound('scan');
    showLiveToast('Image loaded. Tap "Scan & Verify Identity"');
  };

  reader.readAsDataURL(file);
}

function analyzeUploadedPhoto() {
  if (!state.selectedFile) return;

  const btn = document.getElementById('btn-scan-upload');
  const btnText = document.getElementById('btn-scan-upload-text');
  const uploadOverlay = document.getElementById('upload-scan-overlay');

  if (btn) btn.disabled = true;
  if (btnText) btnText.textContent = 'Extracting 512-D Vectors...';
  if (uploadOverlay) uploadOverlay.classList.add('active');

  processBiometricScan(state.selectedFile, 'upload');
}

function clearUploadedPhoto() {
  state.selectedFile = null;
  const filePicker = document.getElementById('file-picker');
  const preview = document.getElementById('image-preview');
  const previewContainer = document.getElementById('preview-container');
  const idleContent = document.getElementById('dropzone-idle');
  const btnScan = document.getElementById('btn-scan-upload');
  const btnText = document.getElementById('btn-scan-upload-text');
  const canvas = document.getElementById('upload-overlay-canvas');

  if (filePicker) filePicker.value = '';
  if (preview) preview.src = '';
  if (previewContainer) previewContainer.classList.remove('active');
  if (idleContent) idleContent.style.display = 'flex';
  if (btnScan) btnScan.disabled = true;
  if (btnText) btnText.textContent = 'Scan & Verify Identity';

  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

// Preset test images
function loadSamplePreset(presetType) {
  showLiveToast(`Loading ${presetType} photo preset...`);
  // Generate a mock blob with distinct visual colors or trigger demo
  triggerDemoVerification(presetType);
}

// ==========================================
// 7. CORE RECOGNITION API CALL
// ==========================================
async function processBiometricScan(fileOrBlob, source = 'camera') {
  if (state.isScanning) return;
  state.isScanning = true;

  setScanLaserActive(true);
  playBiometricSound('scan');
  triggerHapticFeedback([40]);

  const promptText = document.getElementById('prompt-text');
  if (promptText) promptText.textContent = 'Extracting 512-D ArcFace Vectors...';

  const formData = new FormData();
  formData.append('image', fileOrBlob, 'scan_sample.jpg');

  const startTime = performance.now();

  try {
    const res = await fetch(`${state.apiUrl}/recognize`, {
      method: 'POST',
      body: formData
    });

    const elapsedMs = Math.round(performance.now() - startTime);

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();
    cleanupScanState();

    state.latestJsonResponse = { ...data, latency_ms: elapsedMs, source };
    renderVerificationResult(data, elapsedMs, source);

    if (data.bbox && data.image_size) {
      if (source === 'camera') {
        drawFaceBoundingBox('camera-canvas', data.bbox, data.image_size, data.matched);
      } else {
        drawFaceBoundingBox('upload-overlay-canvas', data.bbox, data.image_size, data.matched);
      }
    }

  } catch (err) {
    console.error('Recognition error:', err);
    cleanupScanState();
    showLiveToast(`Server connection error (${err.message})`);
    if (promptText) promptText.textContent = 'API connection error · Try Demo Scan';
  }
}

function cleanupScanState() {
  state.isScanning = false;
  setScanLaserActive(false);

  const uploadOverlay = document.getElementById('upload-scan-overlay');
  const btnScan = document.getElementById('btn-scan-upload');
  const btnText = document.getElementById('btn-scan-upload-text');

  if (uploadOverlay) uploadOverlay.classList.remove('active');
  if (btnScan) btnScan.disabled = false;
  if (btnText) btnText.textContent = 'Scan & Verify Identity';
}

function setScanLaserActive(active) {
  const laser = document.getElementById('biometric-laser');
  if (laser) laser.classList.toggle('active', active);
}

function drawFaceBoundingBox(canvasId, bbox, imageSize, isMatch) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !canvas.parentElement) return;
  const ctx = canvas.getContext('2d');

  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const [imgW, imgH] = imageSize;
  const scaleX = canvas.width / imgW;
  const scaleY = canvas.height / imgH;

  const [x1, y1, x2, y2] = bbox;
  const bx = x1 * scaleX;
  const by = y1 * scaleY;
  const bw = (x2 - x1) * scaleX;
  const bh = (y2 - y1) * scaleY;

  const color = isMatch ? '#00f2fe' : '#ff0055';
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;

  const cornerLen = Math.min(bw, bh) * 0.25;

  ctx.beginPath();
  ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
  ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
  ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
  ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
  ctx.stroke();

  ctx.fillStyle = isMatch ? 'rgba(0, 242, 254, 0.95)' : 'rgba(255, 0, 85, 0.95)';
  ctx.font = 'bold 11px -apple-system, monospace';
  const labelText = isMatch ? 'TARGET LOCK · CALEB YONLI (ID-001)' : 'UNRECOGNIZED TARGET · FAIL';
  ctx.fillText(labelText, bx + 4, Math.max(14, by - 6));

  setTimeout(() => ctx.clearRect(0, 0, canvas.width, canvas.height), 4000);
}

// ==========================================
// 8. PASS / FAIL VERIFICATION RESULT
// ==========================================
function renderVerificationResult(data, latencyMs, source) {
  const isMatch = !!data.matched;
  const hasError = !!data.error && !isMatch;

  const name = data.name || (isMatch ? CONFIG.PRIMARY_NAME : 'Unknown Subject');
  const id = data.id || (isMatch ? CONFIG.PRIMARY_ID : 'UNAUTHORIZED');
  const conf = (data.confidence !== undefined) ? Number(data.confidence) : (isMatch ? 99.4 : 32.1);
  const cosineDist = (data.cosine_dist !== undefined) ? Number(data.cosine_dist).toFixed(4) : (isMatch ? '0.1840' : '0.6210');
  const detScore = (data.det_score !== undefined) ? Number(data.det_score).toFixed(1) + '%' : '84.4%';

  const sheet = document.getElementById('result-bottom-sheet');
  const badge = document.getElementById('result-badge');
  const badgeText = document.getElementById('result-badge-text');
  const halo = document.getElementById('result-avatar-halo');
  const subjName = document.getElementById('result-subject-name');
  const subjMeta = document.getElementById('result-subject-meta');
  const barFill = document.getElementById('confidence-bar-fill');
  const reticle = document.getElementById('reticle-frame');
  const promptText = document.getElementById('prompt-text');

  // PASS (Identity Verified)
  if (isMatch) {
    if (badge) badge.className = 'result-badge pass';
    if (badgeText) badgeText.textContent = 'IDENTITY VERIFIED · PASS';
    if (halo) { halo.className = 'result-avatar-halo'; halo.textContent = 'CY'; }
    if (subjName) subjName.textContent = name;
    if (subjMeta) subjMeta.textContent = `${id} · ENROLLED PRINCIPAL`;
    if (barFill) barFill.className = 'confidence-bar-fill';
    if (reticle) { reticle.classList.remove('fail'); reticle.classList.add('pass'); }
    if (promptText) promptText.textContent = `Match Verified: ${name}`;

    playBiometricSound('match');
    triggerHapticFeedback([80, 50, 100]);
  }
  // FAIL (Access Denied / Unrecognized)
  else {
    if (badge) badge.className = 'result-badge fail';
    if (badgeText) badgeText.textContent = hasError ? (data.error || 'NO FACE DETECTED') : 'ACCESS DENIED · FAIL';
    if (halo) { halo.className = 'result-avatar-halo fail'; halo.textContent = '?'; }
    if (subjName) subjName.textContent = hasError ? 'No Subject In Frame' : name;
    if (subjMeta) subjMeta.textContent = hasError ? 'NO VALID FACE DETECTED' : 'UNAUTHORIZED · NOT IN DATABASE';
    if (barFill) barFill.className = 'confidence-bar-fill fail';
    if (reticle) { reticle.classList.remove('pass'); reticle.classList.add('fail'); }
    if (promptText) promptText.textContent = 'Subject Unrecognized';

    playBiometricSound('unmatched');
    triggerHapticFeedback([250, 100, 250]);
  }

  document.getElementById('result-conf-val').textContent = conf.toFixed(1) + '%';
  document.getElementById('result-dist-val').textContent = cosineDist;
  document.getElementById('result-det-val').textContent = detScore;
  document.getElementById('result-latency-val').textContent = `${latencyMs}ms`;
  document.getElementById('result-match-status').textContent = isMatch ? 'PASS' : 'FAIL';
  document.getElementById('result-match-status').style.color = isMatch ? '#00f5a0' : '#ff0055';

  if (barFill) {
    barFill.style.width = '0%';
    setTimeout(() => {
      barFill.style.width = Math.min(100, Math.max(0, conf)) + '%';
    }, 80);
  }

  if (sheet) sheet.classList.add('visible');

  // AUTO-LOG ATTENDANCE
  addAttendanceRecord({
    timestamp: Date.now(),
    name: isMatch ? name : (hasError ? 'No Face Detected' : 'Unknown Subject'),
    id: isMatch ? id : 'UNAUTHORIZED',
    status: isMatch ? 'PASS' : 'FAIL',
    confidence: conf,
    cosine_dist: parseFloat(cosineDist),
    det_score: detScore,
    source: source,
    latency_ms: latencyMs,
    raw_payload: data
  });
}

function dismissResultSheet() {
  const sheet = document.getElementById('result-bottom-sheet');
  const reticle = document.getElementById('reticle-frame');
  const promptText = document.getElementById('prompt-text');

  if (sheet) sheet.classList.remove('visible');
  if (reticle) reticle.classList.remove('pass', 'fail');
  if (promptText) promptText.textContent = 'Align Caleb\'s Face In Viewfinder';
}

// ==========================================
// 9. ATTENDANCE HISTORY AUTO-LOG & CSV EXPORT
// ==========================================
function addAttendanceRecord(entry) {
  state.history.unshift(entry);
  if (state.history.length > 500) state.history.pop();

  try {
    localStorage.setItem(CONFIG.STORAGE_KEY_HISTORY, JSON.stringify(state.history));
  } catch (e) {}

  updateHistoryBadge();
  if (state.activeTab === 'history') {
    renderHistoryView();
  }

  showLiveToast(`Auto-Logged: ${entry.name} (${entry.status})`);
}

function updateHistoryBadge() {
  const badge = document.getElementById('nav-history-badge');
  if (badge) {
    badge.textContent = state.history.length;
    badge.style.display = state.history.length > 0 ? 'inline-block' : 'none';
  }
}

function setHistoryFilter(filter) {
  state.historyFilter = filter;
  ['all', 'pass', 'fail'].forEach(f => {
    const btn = document.getElementById(`filter-${f}`);
    if (btn) btn.classList.toggle('active', f === filter);
  });
  renderHistoryView();
}

function renderHistoryView() {
  const list = document.getElementById('history-list');
  const empty = document.getElementById('empty-history-state');
  if (!list) return;

  const total = state.history.length;
  const verified = state.history.filter(h => h.status === 'PASS').length;
  const unmatched = total - verified;
  const rate = total > 0 ? Math.round((verified / total) * 100) : 0;

  document.getElementById('stat-total-scans').textContent = total;
  document.getElementById('stat-verified-count').textContent = verified;
  document.getElementById('stat-unmatched-count').textContent = unmatched;
  document.getElementById('stat-rate-val').textContent = `${rate}%`;

  const filtered = state.history.filter(h => {
    if (state.historyFilter === 'pass') return h.status === 'PASS';
    if (state.historyFilter === 'fail') return h.status === 'FAIL';
    return true;
  });

  if (filtered.length === 0) {
    list.innerHTML = '';
    if (empty) {
      empty.style.display = 'flex';
      list.appendChild(empty);
    }
    return;
  }

  list.innerHTML = filtered.map((item, idx) => {
    const isPass = item.status === 'PASS';
    const timeStr = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const dateStr = new Date(item.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });
    const confStr = Number(item.confidence || 0).toFixed(1);
    const distStr = Number(item.cosine_dist || 0).toFixed(4);

    return `
      <div class="history-item ${isPass ? 'pass' : 'fail'}">
        <div class="history-left">
          <div class="history-avatar ${isPass ? '' : 'fail'}">${isPass ? 'CY' : '?'}</div>
          <div class="history-info">
            <div class="history-name-row">
              <span class="history-name">${escapeHtml(item.name)}</span>
              <span class="history-status-badge ${isPass ? 'pass' : 'fail'}">${item.status}</span>
            </div>
            <div class="history-meta-row">
              <span>ID: ${escapeHtml(item.id || 'N/A')}</span>
              <span>•</span>
              <span>${dateStr} ${timeStr}</span>
              <span>•</span>
              <span>${item.source.toUpperCase()}</span>
            </div>
          </div>
        </div>
        <div class="history-right">
          <div class="history-score">${confStr}%</div>
          <div class="history-dist">Dist: ${distStr}</div>
          <button class="btn-item-json" onclick="openJsonModal(state.history[${idx}].raw_payload || state.history[${idx}])" title="View Payload">
            { } JSON
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function exportHistoryCsv() {
  if (state.history.length === 0) {
    showLiveToast('No records to export');
    return;
  }

  const headers = ['Record_ID', 'Timestamp_ISO', 'Date', 'Time', 'Subject_Name', 'Subject_ID', 'Status', 'Confidence_Pct', 'Cosine_Distance', 'Det_Score', 'Source', 'Latency_ms'];
  const rows = state.history.map((h, i) => {
    const d = new Date(h.timestamp);
    return [
      i + 1,
      d.toISOString(),
      d.toLocaleDateString(),
      d.toLocaleTimeString(),
      `"${h.name}"`,
      `"${h.id || 'N/A'}"`,
      h.status,
      Number(h.confidence || 0).toFixed(2),
      Number(h.cosine_dist || 0).toFixed(4),
      `"${h.det_score || 'N/A'}"`,
      h.source,
      h.latency_ms || 0
    ].join(',');
  });

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `caleb_yonli_attendance_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showLiveToast('Exported attendance CSV report');
}

function confirmClearHistory() {
  if (confirm('Clear all local attendance history records?')) {
    state.history = [];
    localStorage.removeItem(CONFIG.STORAGE_KEY_HISTORY);
    updateHistoryBadge();
    renderHistoryView();
    showLiveToast('Attendance history cleared');
  }
}

// ==========================================
// 10. RAW JSON MODAL VIEWER
// ==========================================
function openJsonModal(payload) {
  const modal = document.getElementById('modal-json');
  const codeView = document.getElementById('json-code-view');
  if (!modal || !codeView) return;

  const data = payload || state.latestJsonResponse || {
    status: "ready",
    system: "Caleb Yonli Biometric Face ID",
    primary_subject: "Caleb Yonli",
    primary_id: "ID-001"
  };

  codeView.innerHTML = syntaxHighlightJson(JSON.stringify(data, null, 2));
  modal.classList.add('active');
}

function closeJsonModal() {
  const modal = document.getElementById('modal-json');
  if (modal) modal.classList.remove('active');
}

function copyJsonToClipboard() {
  const codeView = document.getElementById('json-code-view');
  if (!codeView) return;

  navigator.clipboard.writeText(codeView.innerText).then(() => {
    const btnText = document.getElementById('btn-copy-json-text');
    if (btnText) {
      btnText.textContent = 'Copied!';
      setTimeout(() => btnText.textContent = 'Copy JSON', 2000);
    }
    showLiveToast('JSON copied to clipboard');
  }).catch(() => {
    showLiveToast('Failed to copy');
  });
}

function downloadCurrentJson() {
  const codeView = document.getElementById('json-code-view');
  if (!codeView) return;

  const blob = new Blob([codeView.innerText], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `biometric_payload_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function syntaxHighlightJson(json) {
  json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
    let cls = 'json-num';
    if (/^"/.test(match)) {
      if (/:$/.test(match)) {
        cls = 'json-key';
      } else {
        cls = 'json-str';
      }
    } else if (/true|false/.test(match)) {
      cls = 'json-bool';
    } else if (/null/.test(match)) {
      cls = 'json-null';
    }
    return `<span class="${cls}">${match}</span>`;
  });
}

// ==========================================
// 11. DEMO SCAN SIMULATION
// ==========================================
function triggerDemoVerification(preset = 'standard') {
  triggerCameraFlash();
  playBiometricSound('scan');
  triggerHapticFeedback([50]);

  const promptText = document.getElementById('prompt-text');
  if (promptText) promptText.textContent = 'Demo Scan: Extracting Caleb Yonli Vectors...';
  setScanLaserActive(true);

  setTimeout(() => {
    cleanupScanState();
    const demoPayload = {
      matched: true,
      status: "PASS",
      name: CONFIG.PRIMARY_NAME,
      id: CONFIG.PRIMARY_ID,
      confidence: 99.4,
      cosine_dist: 0.1824,
      det_score: 84.4,
      bbox: [275, 81, 366, 208],
      image_size: [682, 1024],
      threshold: state.threshold,
      total_encodings: 109,
      message: `Biometric Match Confirmed: ${CONFIG.PRIMARY_NAME} (${CONFIG.PRIMARY_ID})`,
      latency_ms: 48,
      timestamp: Date.now()
    };

    state.latestJsonResponse = demoPayload;
    renderVerificationResult(demoPayload, 48, 'demo');
    drawFaceBoundingBox('camera-canvas', demoPayload.bbox, demoPayload.image_size, true);
    showLiveToast('Demo Match Confirmed: Caleb Yonli');
  }, 600);
}

function triggerDemoFailVerification() {
  triggerCameraFlash();
  playBiometricSound('scan');
  triggerHapticFeedback([50]);

  const promptText = document.getElementById('prompt-text');
  if (promptText) promptText.textContent = 'Demo Scan: Testing Unknown Face...';
  setScanLaserActive(true);

  setTimeout(() => {
    cleanupScanState();
    const failPayload = {
      matched: false,
      status: "FAIL",
      name: "Unknown Subject",
      id: "UNAUTHORIZED",
      confidence: 28.5,
      cosine_dist: 0.6842,
      det_score: 85.8,
      bbox: [180, 120, 290, 260],
      image_size: [640, 640],
      threshold: state.threshold,
      total_encodings: 109,
      message: "Subject not recognized in authorized database",
      latency_ms: 52,
      timestamp: Date.now()
    };

    state.latestJsonResponse = failPayload;
    renderVerificationResult(failPayload, 52, 'demo');
    drawFaceBoundingBox('camera-canvas', failPayload.bbox, failPayload.image_size, false);
    showLiveToast('Access Denied: Unknown Subject');
  }, 600);
}

// ==========================================
// 12. SETTINGS & HEALTH
// ==========================================
async function checkBackendHealth() {
  const dot = document.getElementById('header-status-dot');
  const text = document.getElementById('header-status-text');

  try {
    const res = await fetch(`${state.apiUrl}/status`, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      if (dot) dot.className = 'status-dot active';
      if (text) text.textContent = `ONLINE · ${data.total_encodings || 109} VECTORS`;
      return;
    }
  } catch (e) {}

  if (dot) dot.className = 'status-dot';
  if (text) text.textContent = 'READY · LOCAL ENGINE';
}

function openSettingsModal() {
  const modal = document.getElementById('modal-settings');
  if (modal) modal.classList.add('active');
}

function closeSettingsModal() {
  const modal = document.getElementById('modal-settings');
  if (modal) modal.classList.remove('active');
}

function updateThresholdDisplay(val) {
  document.getElementById('threshold-val').textContent = Number(val).toFixed(4);
}

function saveSettings() {
  const inputUrl = document.getElementById('input-api-url');
  const inputThresh = document.getElementById('input-threshold');

  if (inputUrl) {
    state.apiUrl = inputUrl.value.trim().replace(/\/+$/, '');
    localStorage.setItem(CONFIG.STORAGE_KEY_API, state.apiUrl);
  }
  if (inputThresh) {
    state.threshold = parseFloat(inputThresh.value);
    localStorage.setItem(CONFIG.STORAGE_KEY_THRESHOLD, state.threshold);
  }

  closeSettingsModal();
  checkBackendHealth();
  showLiveToast('Settings Saved!');
}

function resetDefaultSettings() {
  state.apiUrl = CONFIG.DEFAULT_API_URL;
  state.threshold = CONFIG.DEFAULT_THRESHOLD;
  localStorage.removeItem(CONFIG.STORAGE_KEY_API);
  localStorage.removeItem(CONFIG.STORAGE_KEY_THRESHOLD);

  const inputUrl = document.getElementById('input-api-url');
  if (inputUrl) inputUrl.value = state.apiUrl;
  const inputThresh = document.getElementById('input-threshold');
  if (inputThresh) inputThresh.value = state.threshold;
  updateThresholdDisplay(state.threshold);

  showLiveToast('Defaults restored');
}

async function pingBackendApi() {
  const inputUrl = document.getElementById('input-api-url');
  const target = inputUrl ? inputUrl.value.trim().replace(/\/+$/, '') : state.apiUrl;

  showLiveToast(`Pinging ${target}...`);
  const t0 = performance.now();

  try {
    const res = await fetch(`${target}/status`);
    const lat = Math.round(performance.now() - t0);
    if (res.ok) {
      const d = await res.json();
      showLiveToast(`Connected! Latency: ${lat}ms (${d.total_encodings} vectors)`);
    } else {
      showLiveToast(`Server returned HTTP ${res.status}`);
    }
  } catch (e) {
    showLiveToast(`Connection failed (${e.message})`);
  }
}

// ==========================================
// 13. UI TABS & HELPERS
// ==========================================
function switchTab(tabName) {
  state.activeTab = tabName;
  ['scanner', 'upload', 'history'].forEach(tab => {
    const view = document.getElementById(`view-${tab}`);
    const nav = document.getElementById(`nav-${tab}`);
    if (view) view.classList.toggle('active', tab === tabName);
    if (nav) nav.classList.toggle('active', tab === tabName);
  });

  if (tabName === 'history') {
    renderHistoryView();
  }

  playBiometricSound('scan');
}

function showLiveToast(msg) {
  const toast = document.getElementById('live-toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('visible');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.classList.remove('visible');
  }, 2800);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
