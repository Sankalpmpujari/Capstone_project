/**
 * NeuroBreathe AI - Clinical Diagnostic Controller
 * Implements real-time PCM WAV audio recording, Maximum Phonation Time (MPT) calculation,
 * Chart.js spectral telemetry, preset calibration, and PredictCard rendering.
 */

const state = {
  currentTab: 'dashboard',
  
  // Respiratory Recording
  respRecording: false,
  respMediaStream: null,
  respProcessor: null,
  respPcmSamples: [],
  respPcmBlob: null,
  respAudioContext: null,
  respAnalyser: null,
  respAnimId: null,
  respStartTime: null,
  respTimerInterval: null,
  respSelectedFile: null,
  
  // Parkinson Phonation (MPT) Recording
  parkRecording: false,
  parkMediaStream: null,
  parkProcessor: null,
  parkPcmSamples: [],
  parkPcmBlob: null,
  parkAudioContext: null,
  parkAnalyser: null,
  parkAnimId: null,
  parkStartTime: null,
  parkTimerInterval: null,
  parkSelectedFile: null,

  // Multi-Modal Composite Diagnostics State
  respResults: {
    auscultation: null,
    spirometry: null,
    vitals: null
  },
  parkResults: {
    vocal: null,
    spiral: null
  },
  respComposite: null,
  parkComposite: null,

  // Visualizations
  respMfccChart: null,
  batchResults: null
};

document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  await checkHealth();
  loadParkinsonPreset('healthy_female', false);

  // Close doctor report modal when clicking on backdrop or Esc key
  const modal = document.getElementById('report-modal');
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeDoctorReport();
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDoctorReport();
  });

  // Setup Respiratory Audio Dropzone drag-and-drop
  const dropzone = document.getElementById('resp-audio-dropzone');
  if (dropzone) {
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dropzone-active');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dropzone-active');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleAudioFileUpload(dt.files[0]);
      }
    });
  }
});

// Tab Routing
function switchTab(tabId) {
  // If user clicks a sub-test link, seamlessly map to parent Center and filter station
  let targetStation = null;
  if (tabId === 'spiral') {
    tabId = 'parkinson';
    targetStation = 'spiral';
  } else if (tabId === 'spirometry') {
    tabId = 'respiratory';
    targetStation = 'spiro';
  } else if (tabId === 'rppg') {
    tabId = 'respiratory';
    targetStation = 'rppg';
  }

  // Stop camera if leaving respiratory / rppg
  if (tabId !== 'respiratory' && window.rppgScanning) {
    stopRppgScan();
  }

  state.currentTab = tabId;

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });

  const activeBtn = document.getElementById(`nav-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.add('active');
  }

  const tabs = ['dashboard', 'respiratory', 'parkinson', 'batch', 'models-info'];
  tabs.forEach(t => {
    const el = document.getElementById(`tab-${t}`);
    if (el) {
      el.classList.toggle('hidden', t !== tabId);
    }
  });

  if (tabId === 'parkinson') {
    setTimeout(initSpiralCanvas, 80);
    if (targetStation) {
      setTimeout(() => filterParkStation(targetStation), 60);
    }
  } else if (tabId === 'respiratory' && targetStation) {
    setTimeout(() => filterRespStation(targetStation), 60);
  }

  if (window.lucide) {
    lucide.createIcons();
  }
}

// System Health Verification
async function checkHealth() {
  const badge = document.getElementById('system-status-badge');
  const text = document.getElementById('status-text');

  try {
    const res = await fetch('/api/health');
    const data = await res.json();

    if (data.status === 'healthy') {
      badge.className = 'flex items-center gap-2 px-2.5 py-1 rounded bg-green-950/60 border border-green-800 text-green-400 text-xs font-medium';
      text.textContent = 'Diagnostic models online';
    } else {
      badge.className = 'flex items-center gap-2 px-2.5 py-1 rounded bg-amber-950/60 border border-amber-800 text-amber-400 text-xs font-medium';
      text.textContent = 'Models degraded';
    }
  } catch (err) {
    badge.className = 'flex items-center gap-2 px-2.5 py-1 rounded bg-red-950/60 border border-red-800 text-red-400 text-xs font-medium';
    text.textContent = 'Server offline';
  }
}

// =====================================================================
// UTILITY: Standards-Compliant 16-Bit PCM WAV Encoder
// Converts any Float32Array PCM samples directly into a pristine uncompressed WAV blob.
// =====================================================================
function encodePcmWav(float32Array, sampleRate) {
  const numChannels = 1;
  const bitDepth = 16;
  const dataLength = float32Array.length * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM Format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
  view.setUint16(32, numChannels * (bitDepth / 8), true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

// Global cross-compatibility aliases
window.encodePcmWav = encodePcmWav;
window.encodeWAV = encodePcmWav;

function audioBufferToPcmWav(audioBuffer) {
  return encodePcmWav(audioBuffer.getChannelData(0), audioBuffer.sampleRate);
}

// Decode raw recorded container chunks into pure PCM WAV with error resilience
async function convertChunksToWavBlob(chunks) {
  const rawBlob = new Blob(chunks);
  const arrayBuffer = await rawBlob.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const wavBlob = audioBufferToPcmWav(audioBuffer);
  audioContext.close();
  return wavBlob;
}

// =====================================================================
// MODULE 1: RESPIRATORY ACOUSTIC SCREENING
// =====================================================================

// 1-Click Clinical Sample Analysis
async function selectAndAnalyzeSample(sampleId) {
  const timeBadge = document.getElementById('resp-badge-timestamp');
  timeBadge.textContent = 'Analyzing sample...';

  try {
    const res = await fetch(`/api/respiratory/predict-sample/${sampleId}`, { method: 'POST' });
    const json = await res.json();

    if (json.status !== 'success') {
      throw new Error(json.detail || 'Analysis error');
    }

    renderRespiratoryResults(json.data, `Sample: ${json.filename}`);
  } catch (err) {
    alert(`Respiratory analysis error: ${err.message}`);
    timeBadge.textContent = 'Analysis halted';
  }
}

// File Upload Handler (Supports both file input change event and drag-and-drop File object)
function handleAudioFileUpload(eventOrFile) {
  let file = null;
  if (eventOrFile instanceof File) {
    file = eventOrFile;
  } else if (eventOrFile && eventOrFile.target && eventOrFile.target.files) {
    file = eventOrFile.target.files[0];
  } else if (eventOrFile && eventOrFile.dataTransfer && eventOrFile.dataTransfer.files) {
    file = eventOrFile.dataTransfer.files[0];
  }
  if (!file) return;

  state.respSelectedFile = file;

  const label = document.getElementById('uploaded-file-name');
  if (label) {
    label.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    label.classList.remove('hidden');
  }

  const previewBox = document.getElementById('uploaded-file-preview-box');
  if (previewBox) {
    previewBox.classList.remove('hidden');
  }

  const audioPlayback = document.getElementById('uploaded-audio-playback');
  if (audioPlayback) {
    audioPlayback.src = URL.createObjectURL(file);
    audioPlayback.classList.remove('hidden');
  }

  const btn = document.getElementById('btn-analyze-uploaded');
  if (btn) {
    btn.classList.remove('hidden');
  }

  if (window.lucide) {
    lucide.createIcons();
  }
}

async function submitUploadedAudio() {
  if (!state.respSelectedFile) {
    alert('Please select or drop an audio file first.');
    return;
  }

  const btn = document.getElementById('btn-analyze-uploaded');
  if (btn) {
    btn.innerHTML = '<span class="inline-block animate-spin h-3.5 w-3.5 border-2 border-white border-t-transparent rounded-full mr-1.5"></span> Analyzing audio...';
    btn.disabled = true;
  }

  const formData = new FormData();
  formData.append('file', state.respSelectedFile);

  try {
    const res = await fetch('/api/respiratory/predict-audio', {
      method: 'POST',
      body: formData
    });
    const json = await res.json();
    if (json.status !== 'success') {
      throw new Error(json.detail || 'Audio analysis failed');
    }
    renderRespiratoryResults(json.data, state.respSelectedFile.name);
  } catch (err) {
    alert(`Analysis error: ${err.message}`);
  } finally {
    if (btn) {
      btn.innerHTML = '<i data-lucide="activity" class="h-3.5 w-3.5 mr-1.5 inline"></i> Analyze Lung Sound Recording';
      btn.disabled = false;
      if (window.lucide) lucide.createIcons();
    }
  }
}

// Live Stethoscope / Microphone Recording
async function toggleRespRecording() {
  const btn = document.getElementById('btn-resp-record-toggle');
  const label = document.getElementById('btn-resp-record-label');
  const timerEl = document.getElementById('resp-recording-timer');
  const msg = document.getElementById('resp-mic-idle-msg');

  if (!state.respRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });
      state.respMediaStream = stream;
      state.respPcmSamples = [];

      state.respAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (state.respAudioContext.state === 'suspended') {
        await state.respAudioContext.resume();
      }

      const source = state.respAudioContext.createMediaStreamSource(stream);
      state.respAnalyser = state.respAudioContext.createAnalyser();
      state.respAnalyser.fftSize = 512;
      source.connect(state.respAnalyser);

      // Direct float32 PCM buffer collection without lossy container compression
      const bufferSize = 4096;
      state.respProcessor = state.respAudioContext.createScriptProcessor(bufferSize, 1, 1);
      state.respProcessor.onaudioprocess = (e) => {
        if (!state.respRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        state.respPcmSamples.push(new Float32Array(channelData));
      };
      source.connect(state.respProcessor);
      state.respProcessor.connect(state.respAudioContext.destination);

      drawOscilloscope('resp-audio-canvas', state.respAnalyser);
      msg.classList.add('hidden');

      state.respRecording = true;
      state.respStartTime = Date.now();

      btn.classList.add('bg-red-700', 'text-white');
      btn.classList.remove('bg-slate-800');
      label.textContent = 'Stop recording';

      state.respTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.respStartTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        timerEl.textContent = `${mins}:${secs}`;
      }, 500);

    } catch (err) {
      alert(`Microphone access error: ${err.message}. Please allow microphone permissions.`);
    }
  } else {
    // Stop recording
    state.respRecording = false;
    clearInterval(state.respTimerInterval);
    cancelAnimationFrame(state.respAnimId);

    if (state.respProcessor) {
      state.respProcessor.disconnect();
      state.respProcessor = null;
    }
    if (state.respMediaStream) {
      state.respMediaStream.getTracks().forEach(t => t.stop());
      state.respMediaStream = null;
    }

    const totalSamples = state.respPcmSamples.reduce((sum, chunk) => sum + chunk.length, 0);
    if (totalSamples > 0) {
      const merged = new Float32Array(totalSamples);
      let offset = 0;
      for (const chunk of state.respPcmSamples) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      const sampleRate = state.respAudioContext ? state.respAudioContext.sampleRate : 44100;
      state.respPcmBlob = encodePcmWav(merged, sampleRate);

      const audioUrl = URL.createObjectURL(state.respPcmBlob);
      const player = document.getElementById('resp-recorded-playback');
      player.src = audioUrl;
      player.classList.remove('hidden');
      document.getElementById('btn-analyze-resp-recorded').classList.remove('hidden');

      msg.classList.remove('hidden');
      msg.textContent = `Captured ${(totalSamples / sampleRate).toFixed(1)}s uncompressed PCM WAV. Ready for analysis.`;
    } else {
      msg.classList.remove('hidden');
      msg.textContent = 'Recording stopped. No audio frames captured.';
    }

    if (state.respAudioContext) {
      state.respAudioContext.close();
      state.respAudioContext = null;
    }

    btn.classList.remove('bg-red-700');
    btn.classList.add('bg-slate-800');
    label.textContent = 'Record new sample';
  }
}

// Analyze Recorded Lung Sound
async function analyzeRecordedRespAudio() {
  if (!state.respPcmBlob) {
    alert('No recording found.');
    return;
  }

  const btn = document.getElementById('btn-analyze-resp-recorded');
  btn.textContent = 'Processing acoustic features...';
  btn.disabled = true;

  const formData = new FormData();
  formData.append('file', state.respPcmBlob, 'recorded_lung_sound.wav');
  formData.append('is_live_mic', 'true');

  try {
    const res = await fetch('/api/respiratory/predict-audio', {
      method: 'POST',
      body: formData
    });
    const json = await res.json();
    if (json.status !== 'success') {
      throw new Error(json.detail || 'Analysis error');
    }
    renderRespiratoryResults(json.data, 'Live Microphone Recording');
  } catch (err) {
    alert(`Failed to analyze audio: ${err.message}`);
  } finally {
    btn.textContent = 'Analyze lung sound';
    btn.disabled = false;
  }
}

// Render Respiratory PredictCard
function renderRespiratoryResults(data, sourceLabel) {
  document.getElementById('resp-placeholder').classList.add('hidden');
  document.getElementById('resp-result-details').classList.remove('hidden');

  const banner = document.getElementById('resp-diagnosis-banner');
  const badge = document.getElementById('resp-risk-badge');
  const diagText = document.getElementById('resp-diagnosis-text');
  const confText = document.getElementById('resp-confidence-text');
  const timeBadge = document.getElementById('resp-badge-timestamp');

  timeBadge.textContent = `${sourceLabel}, ${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;

  const isCopd = (data.prediction === 'COPD');
  diagText.textContent = data.diagnosis || (isCopd ? 'Chronic Obstructive Pulmonary Disease (COPD)' : 'Non-COPD Respiratory Profile');
  confText.textContent = `Classification confidence: ${(data.confidence * 100).toFixed(1)}%`;

  const riskColor = data.risk_color || (isCopd ? 'red' : 'emerald');
  if (riskColor === 'red') {
    banner.className = 'p-3.5 rounded border border-red-800/80 bg-red-950/25';
    badge.className = 'px-2 py-0.5 rounded text-xs font-semibold bg-red-950 text-red-300 border border-red-800';
    badge.textContent = data.risk_badge || 'High Risk / Action Required';
  } else if (riskColor === 'amber') {
    banner.className = 'p-3.5 rounded border border-amber-800/80 bg-amber-950/25';
    badge.className = 'px-2 py-0.5 rounded text-xs font-semibold bg-amber-950 text-amber-300 border border-amber-800';
    badge.textContent = data.risk_badge || 'Moderate Suspicion';
  } else {
    banner.className = 'p-3.5 rounded border border-green-800/80 bg-green-950/25';
    badge.className = 'px-2 py-0.5 rounded text-xs font-semibold bg-green-950 text-green-300 border border-green-800';
    badge.textContent = data.risk_badge || 'Normal / Low Risk';
  }

  // Probabilities
  const copdPct = Math.round((data.probabilities.COPD || 0) * 100);
  const notCopdPct = Math.round((data.probabilities.Not_COPD || 0) * 100);

  document.getElementById('resp-copd-pct').textContent = `${copdPct}%`;
  document.getElementById('resp-notcopd-pct').textContent = `${notCopdPct}%`;
  document.getElementById('resp-bar-copd').style.width = `${copdPct}%`;
  document.getElementById('resp-bar-notcopd').style.width = `${notCopdPct}%`;

  // Respiratory Cadence (BPM)
  const rateCard = document.getElementById('resp-rate-card');
  if (data.respiratory_rate && rateCard) {
    rateCard.classList.remove('hidden');
    const bpmEl = document.getElementById('metric-resp-bpm');
    const badgeEl = document.getElementById('metric-resp-bpm-badge');
    const descEl = document.getElementById('metric-resp-tempo-desc');

    bpmEl.textContent = data.respiratory_rate.bpm;
    badgeEl.textContent = data.respiratory_rate.tempo;

    if (data.respiratory_rate.is_fast) {
      badgeEl.className = 'px-2 py-0.5 rounded text-[11px] font-semibold bg-red-950 text-red-300 border border-red-800';
      descEl.textContent = `Rapid breathing detected (${data.respiratory_rate.cycle_duration_sec}s cycle). Normal: 12-20 breaths/min.`;
    } else {
      badgeEl.className = 'px-2 py-0.5 rounded text-[11px] font-semibold bg-green-950 text-green-300 border border-green-800';
      descEl.textContent = `Stable breathing rhythm (${data.respiratory_rate.cycle_duration_sec}s cycle). Normal: 12-20 breaths/min.`;
    }
  }

  // Interpretation
  document.getElementById('resp-interpretation-text').textContent = data.interpretation;

  // Extracted Acoustic Metrics
  if (data.acoustic_metrics) {
    document.getElementById('metric-resp-duration').textContent = `${data.acoustic_metrics.duration_seconds} s`;
    document.getElementById('metric-resp-centroid').textContent = `${data.acoustic_metrics.spectral_centroid_hz} Hz`;
    document.getElementById('metric-resp-rms').textContent = data.acoustic_metrics.rms_energy;
    document.getElementById('metric-resp-zcr').textContent = data.acoustic_metrics.zero_crossing_rate;
  }

  // Waveform Envelope
  if (data.waveform && data.waveform.length > 0) {
    renderWaveformEnvelope('resp-waveform-canvas', data.waveform);
  }

  // MFCC Chart
  if (data.mfcc_summary && data.mfcc_summary.length > 0) {
    renderMfccBarChart(data.mfcc_summary);
  }

  // Update multi-modal respiratory state
  state.respResults.auscultation = data;
  evaluateRespiratoryComposite();

  if (window.lucide) {
    lucide.createIcons();
  }
}

// =====================================================================
// MODULE 2: PARKINSON'S VOICE BIOMARKERS & MPT TEST
// =====================================================================

// Live Phonation Recording & MPT Test
async function toggleParkinsonRecording() {
  const btn = document.getElementById('btn-park-record-toggle');
  const label = document.getElementById('btn-park-record-label');
  const timerEl = document.getElementById('park-recording-timer');
  const msg = document.getElementById('park-mic-idle-msg');

  if (!state.parkRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });
      state.parkMediaStream = stream;
      state.parkPcmSamples = [];

      state.parkAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (state.parkAudioContext.state === 'suspended') {
        await state.parkAudioContext.resume();
      }

      const source = state.parkAudioContext.createMediaStreamSource(stream);
      state.parkAnalyser = state.parkAudioContext.createAnalyser();
      state.parkAnalyser.fftSize = 512;
      source.connect(state.parkAnalyser);

      const bufferSize = 4096;
      state.parkProcessor = state.parkAudioContext.createScriptProcessor(bufferSize, 1, 1);
      state.parkProcessor.onaudioprocess = (e) => {
        if (!state.parkRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        state.parkPcmSamples.push(new Float32Array(channelData));
      };
      source.connect(state.parkProcessor);
      state.parkProcessor.connect(state.parkAudioContext.destination);

      drawOscilloscope('park-audio-canvas', state.parkAnalyser);
      msg.classList.add('hidden');

      state.parkRecording = true;
      state.parkStartTime = Date.now();

      btn.classList.add('bg-red-700', 'text-white');
      btn.classList.remove('bg-slate-800');
      label.textContent = 'Stop phonation test';

      state.parkTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.parkStartTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        timerEl.textContent = `${mins}:${secs}`;
      }, 500);

    } catch (err) {
      alert(`Microphone access error: ${err.message}. Please allow microphone access.`);
    }
  } else {
    // Stop phonation test
    state.parkRecording = false;
    clearInterval(state.parkTimerInterval);
    cancelAnimationFrame(state.parkAnimId);

    if (state.parkProcessor) {
      state.parkProcessor.disconnect();
      state.parkProcessor = null;
    }
    if (state.parkMediaStream) {
      state.parkMediaStream.getTracks().forEach(t => t.stop());
      state.parkMediaStream = null;
    }

    const totalSamples = state.parkPcmSamples.reduce((sum, chunk) => sum + chunk.length, 0);
    if (totalSamples > 0) {
      const merged = new Float32Array(totalSamples);
      let offset = 0;
      for (const chunk of state.parkPcmSamples) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      const sampleRate = state.parkAudioContext ? state.parkAudioContext.sampleRate : 44100;
      state.parkPcmBlob = encodePcmWav(merged, sampleRate);

      const audioUrl = URL.createObjectURL(state.parkPcmBlob);
      const player = document.getElementById('park-recorded-playback');
      player.src = audioUrl;
      player.classList.remove('hidden');
      document.getElementById('btn-analyze-park-recorded').classList.remove('hidden');

      msg.classList.remove('hidden');
      msg.textContent = `Captured ${(totalSamples / sampleRate).toFixed(1)}s uncompressed PCM phonation. Ready for MPT analysis.`;
    } else {
      msg.classList.remove('hidden');
      msg.textContent = 'Recording stopped. No audio frames captured.';
    }

    if (state.parkAudioContext) {
      state.parkAudioContext.close();
      state.parkAudioContext = null;
    }

    btn.classList.remove('bg-red-700');
    btn.classList.add('bg-slate-800');
    label.textContent = 'Record new phonation';
  }
}

// Analyze Recorded Phonation Audio for MPT and Parkinson's
async function analyzeRecordedParkinsonAudio() {
  if (!state.parkPcmBlob) {
    alert('No phonation recording found.');
    return;
  }

  const btn = document.getElementById('btn-analyze-park-recorded');
  btn.textContent = 'Analyzing phonation acoustics & MPT...';
  btn.disabled = true;

  const formData = new FormData();
  formData.append('file', state.parkPcmBlob, 'mpt_phonation.wav');

  try {
    const res = await fetch('/api/parkinson/predict-audio', {
      method: 'POST',
      body: formData
    });
    const json = await res.json();
    if (json.status !== 'success') {
      throw new Error(json.detail || 'Phonation analysis failed');
    }
    renderParkinsonResults(json.data, 'Live Maximum Phonation Time (MPT) Test');
  } catch (err) {
    alert(`Phonation assessment error: ${err.message}`);
  } finally {
    btn.textContent = 'Analyze phonation test';
    btn.disabled = false;
  }
}

const PARKINSON_PRESETS = {
  healthy_female: {
    "MDVP:Fo(Hz)": 197.076,
    "MDVP:Fhi(Hz)": 206.896,
    "MDVP:Flo(Hz)": 192.055,
    "MDVP:Jitter(%)": 0.00289,
    "MDVP:Jitter(Abs)": 0.00001,
    "MDVP:RAP": 0.00166,
    "MDVP:PPQ": 0.00168,
    "Jitter:DDP": 0.00498,
    "MDVP:Shimmer": 0.01098,
    "MDVP:Shimmer(dB)": 0.097,
    "Shimmer:APQ3": 0.00563,
    "Shimmer:APQ5": 0.0068,
    "MDVP:APQ": 0.00802,
    "Shimmer:DDA": 0.01689,
    "NHR": 0.00339,
    "HNR": 26.775,
    "RPDE": 0.422229,
    "DFA": 0.741367,
    "spread1": -7.3483,
    "spread2": 0.177551,
    "D2": 1.743867,
    "PPE": 0.085569
  },
  healthy_male: {
    "MDVP:Fo(Hz)": 198.383,
    "MDVP:Fhi(Hz)": 215.203,
    "MDVP:Flo(Hz)": 193.104,
    "MDVP:Jitter(%)": 0.00212,
    "MDVP:Jitter(Abs)": 0.00001,
    "MDVP:RAP": 0.00113,
    "MDVP:PPQ": 0.00135,
    "Jitter:DDP": 0.00339,
    "MDVP:Shimmer": 0.01263,
    "MDVP:Shimmer(dB)": 0.111,
    "Shimmer:APQ3": 0.0064,
    "Shimmer:APQ5": 0.00825,
    "MDVP:APQ": 0.00951,
    "Shimmer:DDA": 0.01919,
    "NHR": 0.00119,
    "HNR": 30.08,
    "RPDE": 0.34761,
    "DFA": 0.64413,
    "spread1": -7.389537,
    "spread2": 0.120712,
    "D2": 1.82298,
    "PPE": 0.084505
  },
  parkinson_moderate: {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.0313,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    "NHR": 0.02211,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": 0.266482,
    "D2": 2.301442,
    "PPE": 0.284654
  },
  parkinson_severe: {
    "MDVP:Fo(Hz)": 116.676,
    "MDVP:Fhi(Hz)": 137.871,
    "MDVP:Flo(Hz)": 111.366,
    "MDVP:Jitter(%)": 0.00997,
    "MDVP:Jitter(Abs)": 0.00009,
    "MDVP:RAP": 0.00502,
    "MDVP:PPQ": 0.00698,
    "Jitter:DDP": 0.01505,
    "MDVP:Shimmer": 0.05492,
    "MDVP:Shimmer(dB)": 0.517,
    "Shimmer:APQ3": 0.02924,
    "Shimmer:APQ5": 0.04005,
    "MDVP:APQ": 0.03772,
    "Shimmer:DDA": 0.08771,
    "NHR": 0.01353,
    "HNR": 20.644,
    "RPDE": 0.434969,
    "DFA": 0.819235,
    "spread1": -4.117501,
    "spread2": 0.334147,
    "D2": 2.405554,
    "PPE": 0.368975
  }
};

function loadParkinsonPreset(presetKey, autoSubmit = true) {
  const preset = PARKINSON_PRESETS[presetKey];
  if (!preset) return;

  for (const [feat, val] of Object.entries(preset)) {
    const input = document.getElementById(`feat_${feat}`);
    if (input) {
      input.value = val;
    }
  }

  if (autoSubmit) {
    const titles = {
      healthy_female: 'Healthy Control (Female Profile)',
      healthy_male: 'Healthy Control (Male Profile)',
      parkinson_moderate: 'Parkinson Moderate Case',
      parkinson_severe: 'Parkinson Severe Dysphonia'
    };
    submitParkinsonPrediction(`Preset: ${titles[presetKey] || presetKey}`);
  }
}

async function submitParkinsonPrediction(sourceHint = 'Clinical Biomarkers Evaluation') {
  const features = {};
  const featureKeys = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
  ];

  for (const key of featureKeys) {
    const input = document.getElementById(`feat_${key}`);
    if (!input || input.value === '') {
      alert(`Please provide a value for ${key}`);
      return;
    }
    features[key] = parseFloat(input.value);
  }

  try {
    const res = await fetch('/api/parkinson/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    const json = await res.json();
    if (json.status !== 'success') {
      throw new Error(json.detail || 'Prediction failed');
    }
    renderParkinsonResults(json.data, sourceHint);
  } catch (err) {
    alert(`Parkinson assessment error: ${err.message}`);
  }
}

// Render Parkinson PredictCard
function renderParkinsonResults(data, sourceHint) {
  document.getElementById('parkinson-placeholder').classList.add('hidden');
  document.getElementById('parkinson-result-details').classList.remove('hidden');

  const banner = document.getElementById('parkinson-diagnosis-banner');
  const badge = document.getElementById('parkinson-risk-badge');
  const diagText = document.getElementById('parkinson-diagnosis-text');
  const confText = document.getElementById('parkinson-confidence-text');
  const timeBadge = document.getElementById('parkinson-badge-timestamp');

  timeBadge.textContent = `${sourceHint}, ${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;

  const isParkinson = (data.status === 1);
  diagText.textContent = isParkinson ? "Parkinson's Phonation Dysphonia Detected" : "Healthy Vocal Phonation Pattern";
  confText.textContent = `Model confidence: ${(data.confidence * 100).toFixed(1)}%`;

  if (isParkinson) {
    banner.className = 'p-3.5 rounded border border-red-800/80 bg-red-950/20';
    badge.className = 'px-2 py-0.5 rounded text-xs font-medium bg-red-950 text-red-300 border border-red-800';
    badge.textContent = data.risk_level || 'High Risk';
  } else {
    banner.className = 'p-3.5 rounded border border-green-800/80 bg-green-950/20';
    badge.className = 'px-2 py-0.5 rounded text-xs font-medium bg-green-950 text-green-300 border border-green-800';
    badge.textContent = data.risk_level || 'Low Risk / Normal';
  }

  // Probabilities
  const pdPct = Math.round((data.probability_parkinson || 0) * 100);
  const hPct = Math.round((data.probability_healthy || 0) * 100);

  document.getElementById('parkinson-pd-pct').textContent = `${pdPct}%`;
  document.getElementById('parkinson-h-pct').textContent = `${hPct}%`;
  document.getElementById('parkinson-bar-pd').style.width = `${pdPct}%`;
  document.getElementById('parkinson-bar-h').style.width = `${hPct}%`;

  // MPT Telemetry Panel (if available from audio recording)
  const mptBox = document.getElementById('parkinson-mpt-box');
  if (data.mpt) {
    mptBox.classList.remove('hidden');
    document.getElementById('mpt-val-seconds').textContent = `${data.mpt.duration_seconds} s`;
    document.getElementById('mpt-val-status').textContent = data.mpt.status;
    document.getElementById('mpt-val-pitch').textContent = `${data.mpt.mean_pitch_hz} Hz`;
    document.getElementById('mpt-clinical-note').textContent = data.mpt.clinical_note;
  } else {
    mptBox.classList.add('hidden');
  }

  // Clinical Interpretation
  document.getElementById('parkinson-interpretation-text').textContent = data.interpretation;

  // Biomarker Deviation Table
  const tbody = document.getElementById('biomarker-table-body');
  if (tbody) {
    tbody.innerHTML = '';
    (data.biomarker_analysis || []).forEach(item => {
      const tr = document.createElement('tr');

      let statusHtml = '<span class="text-green-400 font-medium">Normal</span>';
      if (item.status === 'elevated') {
        statusHtml = '<span class="text-red-400 font-semibold">Elevated</span>';
      } else if (item.status === 'depressed') {
        statusHtml = '<span class="text-amber-400 font-semibold">Low</span>';
      }

      tr.innerHTML = `
        <td class="text-slate-300">${item.feature}</td>
        <td class="text-slate-100 font-medium">${item.value}</td>
        <td class="text-slate-400">${item.normal_range || '-'}</td>
        <td class="text-right">${statusHtml}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Update multi-modal parkinson state
  state.parkResults.vocal = data;
  evaluateParkinsonComposite();

  if (window.lucide) {
    lucide.createIcons();
  }
}

// =====================================================================
// MODULE 3: BATCH CSV SCREENING
// =====================================================================

function handleBatchFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  state.batchFile = file;
  const label = document.getElementById('batch-file-name');
  label.textContent = `Selected cohort file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  label.classList.remove('hidden');

  document.getElementById('btn-analyze-batch').classList.remove('hidden');
}

async function submitBatchCSV() {
  if (!state.batchFile) {
    alert('Please select a batch CSV file first.');
    return;
  }

  const btn = document.getElementById('btn-analyze-batch');
  btn.textContent = 'Processing cohort dataset...';
  btn.disabled = true;

  const formData = new FormData();
  formData.append('file', state.batchFile);

  try {
    const res = await fetch('/api/parkinson/predict-batch', {
      method: 'POST',
      body: formData
    });
    const json = await res.json();
    if (json.status !== 'success') {
      throw new Error(json.detail || 'Failed to process cohort CSV');
    }
    state.batchResults = json.data;
    renderBatchResults(json.data);
  } catch (err) {
    alert(`Batch processing error: ${err.message}`);
  } finally {
    btn.textContent = 'Process cohort dataset';
    btn.disabled = false;
  }
}

function renderBatchResults(data) {
  const container = document.getElementById('batch-results-container');
  container.classList.remove('hidden');

  document.getElementById('batch-stat-total').textContent = data.summary.total_patients;
  document.getElementById('batch-stat-positive').textContent = data.summary.parkinson_detected;
  document.getElementById('batch-stat-negative').textContent = data.summary.healthy_detected;
  document.getElementById('batch-stat-rate').textContent = `${data.summary.positivity_rate}%`;

  const tbody = document.getElementById('batch-table-body');
  tbody.innerHTML = '';

  data.patients.forEach(patient => {
    const tr = document.createElement('tr');
    const isPd = (patient.prediction === 1);

    let statusText = isPd
      ? '<span class="text-red-400 font-semibold">Parkinson Detected</span>'
      : '<span class="text-green-400">Healthy Control</span>';

    tr.innerHTML = `
      <td class="text-slate-200">${patient.id}</td>
      <td>${statusText}</td>
      <td class="text-slate-300">${(patient.probability_parkinson * 100).toFixed(1)}%</td>
      <td class="text-slate-300">${(patient.probability_healthy * 100).toFixed(1)}%</td>
      <td class="text-right text-slate-300">${patient.risk_level}</td>
    `;
    tbody.appendChild(tr);
  });

  if (window.lucide) {
    lucide.createIcons();
  }
}

function downloadSampleParkinsonCSV() {
  const headers = "name,MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE\n";
  const row1 = "Patient_A,197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,-7.3483,0.177551,1.743867,0.085569\n";
  const row2 = "Patient_B,119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654\n";

  const blob = new Blob([headers + row1 + row2], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'parkinsons_sample_cohort.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function exportBatchResultsCSV() {
  if (!state.batchResults || !state.batchResults.patients) return;

  let csv = "Patient_ID,Diagnosis,Parkinson_Probability,Healthy_Probability,Risk_Level\n";
  state.batchResults.patients.forEach(p => {
    csv += `"${p.id}","${p.diagnosis}",${p.probability_parkinson},${p.probability_healthy},"${p.risk_level}"\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `neurobreathe_cohort_results_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// =====================================================================
// CANVAS DRAWING HELPERS
// =====================================================================

function drawOscilloscope(canvasId, analyser) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !analyser) return;
  const ctx = canvas.getContext('2d');
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  function render() {
    requestAnimationFrame(render);
    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = '#070a12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#0d9488';
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * (canvas.height / 2);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }

  render();
}

function renderWaveformEnvelope(canvasId, points) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const midY = canvas.height / 2;
  const step = canvas.width / points.length;

  ctx.fillStyle = '#0d9488';
  points.forEach((val, i) => {
    const barHeight = Math.max(1.5, val * (canvas.height * 0.85));
    const x = i * step;
    ctx.fillRect(x, midY - barHeight / 2, Math.max(1, step - 1), barHeight);
  });
}

function renderMfccBarChart(mfccValues) {
  const canvas = document.getElementById('resp-mfcc-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (state.respMfccChart) {
    state.respMfccChart.destroy();
  }

  const labels = mfccValues.map((_, i) => `C${i + 1}`);

  state.respMfccChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: mfccValues,
        backgroundColor: '#0d9488',
        borderWidth: 0,
        borderRadius: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { display: false }
        },
        y: {
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { color: '#1e293b' }
        }
      }
    }
  });
}

// =====================================================================
// MODULE 3: DIGITAL ARCHIMEDES SPIRAL & MICROGRAPHIA KINEMATICS
// =====================================================================

let spiralPoints = [];
let isDrawingSpiral = false;
let spiralStartTime = null;
let spiralTimerInterval = null;
let spiralCanvasInitialized = false;

function initSpiralCanvas() {
  const canvas = document.getElementById('spiral-canvas');
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  const width = rect.width > 0 ? Math.floor(rect.width) : 520;
  const height = rect.height > 0 ? Math.floor(rect.height) : 360;
  canvas.width = width;
  canvas.height = height;

  drawSpiralTemplate(canvas);

  if (!spiralCanvasInitialized) {
    spiralCanvasInitialized = true;

    // Mouse handlers
    canvas.addEventListener('mousedown', (e) => startSpiralDrawing(getCanvasPos(canvas, e)));
    canvas.addEventListener('mousemove', (e) => {
      if (isDrawingSpiral) drawSpiralPoint(getCanvasPos(canvas, e));
    });
    window.addEventListener('mouseup', stopSpiralDrawing);

    // Touch handlers
    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      if (e.touches.length > 0) startSpiralDrawing(getCanvasPos(canvas, e.touches[0]));
    }, { passive: false });
    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (isDrawingSpiral && e.touches.length > 0) drawSpiralPoint(getCanvasPos(canvas, e.touches[0]));
    }, { passive: false });
    window.addEventListener('touchend', stopSpiralDrawing);
  }
}

function getCanvasPos(canvas, evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top,
    t: performance.now()
  };
}

function drawSpiralTemplate(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#060913';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;

  // Draw faint reference spiral
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(13, 148, 136, 0.18)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);

  const a = 2.0;
  const b = 3.5;
  const maxTheta = 6.2 * Math.PI;
  for (let theta = 0; theta <= maxTheta; theta += 0.05) {
    const r = a + b * theta;
    const x = cx + r * Math.cos(theta);
    const y = cy + r * Math.sin(theta);
    if (theta === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Center start indicator
  ctx.beginPath();
  ctx.fillStyle = '#2dd4bf';
  ctx.arc(cx, cy, 4.5, 0, 2 * Math.PI);
  ctx.fill();
}

function startSpiralDrawing(pos) {
  isDrawingSpiral = true;
  spiralPoints = [pos];
  spiralStartTime = Date.now();

  const guideMsg = document.getElementById('spiral-guide-msg');
  if (guideMsg) guideMsg.classList.add('hidden');

  const ptsSpan = document.getElementById('spiral-points-count');
  if (ptsSpan) ptsSpan.textContent = '1 pts';

  const timerSpan = document.getElementById('spiral-timer');
  if (spiralTimerInterval) clearInterval(spiralTimerInterval);
  spiralTimerInterval = setInterval(() => {
    if (!spiralStartTime) return;
    const sec = Math.floor((Date.now() - spiralStartTime) / 1000);
    const ms = Math.floor(((Date.now() - spiralStartTime) % 1000) / 100);
    if (timerSpan) timerSpan.textContent = `00:0${sec}.${ms}`;
  }, 100);

  const canvas = document.getElementById('spiral-canvas');
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function drawSpiralPoint(pos) {
  spiralPoints.push(pos);

  const canvas = document.getElementById('spiral-canvas');
  const ctx = canvas.getContext('2d');

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.2;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const prev = spiralPoints[spiralPoints.length - 2];
  if (prev) {
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  }

  const ptsSpan = document.getElementById('spiral-points-count');
  if (ptsSpan) ptsSpan.textContent = `${spiralPoints.length} pts`;
}

function stopSpiralDrawing() {
  if (!isDrawingSpiral) return;
  isDrawingSpiral = false;
  if (spiralTimerInterval) clearInterval(spiralTimerInterval);
}

function clearSpiralCanvas() {
  spiralPoints = [];
  isDrawingSpiral = false;
  spiralStartTime = null;
  if (spiralTimerInterval) clearInterval(spiralTimerInterval);

  const canvas = document.getElementById('spiral-canvas');
  if (canvas) drawSpiralTemplate(canvas);

  const guideMsg = document.getElementById('spiral-guide-msg');
  if (guideMsg) guideMsg.classList.remove('hidden');

  const ptsSpan = document.getElementById('spiral-points-count');
  if (ptsSpan) ptsSpan.textContent = '0 pts';

  const timerSpan = document.getElementById('spiral-timer');
  if (timerSpan) timerSpan.textContent = '00:00';
}

function loadPresetSpiral(type) {
  clearSpiralCanvas();
  const canvas = document.getElementById('spiral-canvas');
  if (!canvas) return;

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const ctx = canvas.getContext('2d');

  const guideMsg = document.getElementById('spiral-guide-msg');
  if (guideMsg) guideMsg.classList.add('hidden');

  spiralPoints = [];
  const startTime = performance.now();

  const totalPoints = 180;
  const maxTheta = 6.2 * Math.PI;

  for (let i = 0; i < totalPoints; i++) {
    const progress = i / totalPoints;
    const theta = progress * maxTheta;
    const t = startTime + progress * 4500; // 4.5s duration

    let r;
    if (type === 'healthy') {
      r = 3.0 + 3.4 * theta + 1.2 * Math.sin(theta * 2.0);
    } else {
      const decayFactor = 1.0 - progress * 0.45;
      const tremor = 3.8 * Math.sin(2 * Math.PI * 5.2 * (progress * 4.5));
      r = Math.max(3.0, (3.0 + 3.2 * theta * decayFactor) + tremor);
    }

    const x = cx + r * Math.cos(theta);
    const y = cy + r * Math.sin(theta);
    spiralPoints.push({ x, y, t });
  }

  ctx.strokeStyle = type === 'healthy' ? '#2dd4bf' : '#f59e0b';
  ctx.lineWidth = 2.2;
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (let i = 0; i < spiralPoints.length; i++) {
    const pt = spiralPoints[i];
    if (i === 0) ctx.moveTo(pt.x, pt.y);
    else ctx.lineTo(pt.x, pt.y);
  }
  ctx.stroke();

  const ptsSpan = document.getElementById('spiral-points-count');
  if (ptsSpan) ptsSpan.textContent = `${spiralPoints.length} pts (preset)`;

  const timerSpan = document.getElementById('spiral-timer');
  if (timerSpan) timerSpan.textContent = '00:04.5';

  analyzeSpiralDrawing();
}

async function analyzeSpiralDrawing() {
  if (spiralPoints.length < 15) {
    alert("Please draw a continuous spiral on the canvas (or load a preset) before analyzing.");
    return;
  }

  const btn = document.getElementById('btn-analyze-spiral');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="animate-spin mr-1">↻</span> Analyzing...';
  }

  try {
    const res = await fetch('/api/parkinson/test-spiral', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points: spiralPoints })
    });

    const result = await res.json();
    if (result.status === 'success') {
      renderSpiralResults(result.data);
    } else {
      alert(result.detail || "Spiral kinematic analysis failed.");
    }
  } catch (err) {
    alert("Analysis failed: " + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i data-lucide="sparkles" class="h-3.5 w-3.5 mr-1.5"></i><span>Analyze Kinematics</span>';
      if (window.lucide) lucide.createIcons();
    }
  }
}

function renderSpiralResults(data) {
  const badge = document.getElementById('spiral-risk-badge');
  if (badge) {
    badge.textContent = data.risk_badge;
    if (data.risk_color === 'red') {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-red-950/80 text-red-400 border border-red-800';
    } else if (data.risk_color === 'amber') {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-amber-950/80 text-amber-400 border border-amber-800';
    } else {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800';
    }
  }

  const scoreVal = document.getElementById('spiral-score-val');
  if (scoreVal) scoreVal.textContent = `${Math.round(data.risk_score * 100)}% (${data.risk_level})`;

  const scoreBar = document.getElementById('spiral-score-bar');
  if (scoreBar) {
    scoreBar.style.width = `${Math.min(100, Math.round(data.risk_score * 100))}%`;
    scoreBar.className = data.risk_color === 'red' ? 'h-full bg-red-500' : (data.risk_color === 'amber' ? 'h-full bg-amber-500' : 'h-full bg-emerald-500');
  }

  const m = data.metrics;
  if (m) {
    const jerkEl = document.getElementById('spiral-metric-jerk');
    if (jerkEl) jerkEl.textContent = m.drawing_jerk;
    const jerkStat = document.getElementById('spiral-jerk-status');
    if (jerkStat) jerkStat.textContent = `Status: ${m.jerk_status}`;

    const decayEl = document.getElementById('spiral-metric-decay');
    if (decayEl) decayEl.textContent = `${m.micrographia_decay_pct}%`;
    const decayStat = document.getElementById('spiral-decay-status');
    if (decayStat) decayStat.textContent = `Pattern: ${m.micrographia_status}`;

    const tremorEl = document.getElementById('spiral-metric-tremor');
    if (tremorEl) tremorEl.textContent = `${m.peak_tremor_hz} Hz (${m.tremor_power_pct}%)`;

    const durEl = document.getElementById('spiral-metric-duration');
    if (durEl) durEl.textContent = `${m.drawing_duration_sec}s`;
    const stallEl = document.getElementById('spiral-stalls-val');
    if (stallEl) stallEl.textContent = `${m.micro_stalls_count} sub-sec stalls`;
  }

  const interp = document.getElementById('spiral-interpretation');
  if (interp) interp.textContent = data.interpretation;

  // Update multi-modal parkinson state
  state.parkResults.spiral = data;
  evaluateParkinsonComposite();
}

// =====================================================================
// MODULE 4: VIRTUAL DYNAMIC ACOUSTIC SPIROMETRY (FEV1 / FVC)
// =====================================================================

let spiroRecording = false;
let spiroMediaStream = null;
let spiroAudioContext = null;
let spiroAnalyser = null;
let spiroProcessor = null;
let spiroPcmSamples = [];
let spiroRecordedBlob = null;
let spiroStartTime = null;
let spiroTimerInterval = null;
let spiroAnimId = null;
let spiroChartInstance = null;

async function toggleSpiroRecording() {
  if (spiroRecording) {
    stopSpiroRecording();
  } else {
    await startSpiroRecording();
  }
}

async function startSpiroRecording() {
  try {
    spiroMediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    alert("Microphone permission denied. Please allow microphone access for forced exhalation analysis.");
    return;
  }

  spiroAudioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = spiroAudioContext.createMediaStreamSource(spiroMediaStream);
  spiroAnalyser = spiroAudioContext.createAnalyser();
  spiroAnalyser.fftSize = 512;
  source.connect(spiroAnalyser);

  spiroPcmSamples = [];
  spiroProcessor = spiroAudioContext.createScriptProcessor(4096, 1, 1);
  spiroProcessor.onaudioprocess = (e) => {
    if (!spiroRecording) return;
    const input = e.inputBuffer.getChannelData(0);
    spiroPcmSamples.push(new Float32Array(input));
  };
  source.connect(spiroProcessor);
  spiroProcessor.connect(spiroAudioContext.destination);

  spiroRecording = true;
  spiroStartTime = Date.now();

  const recordBtn = document.getElementById('btn-spiro-record-toggle');
  const recordLabel = document.getElementById('btn-spiro-record-label');
  if (recordBtn) recordBtn.className = 'px-3.5 py-1.5 rounded bg-red-900/80 hover:bg-red-800 text-red-200 text-xs font-medium border border-red-700 transition-colors flex items-center gap-2';
  if (recordLabel) recordLabel.textContent = 'Blowing... (Auto-stops in 4s)';

  const idleMsg = document.getElementById('spiro-mic-idle-msg');
  if (idleMsg) idleMsg.classList.add('hidden');

  drawSpiroOscilloscope();

  if (spiroTimerInterval) clearInterval(spiroTimerInterval);
  spiroTimerInterval = setInterval(() => {
    const elapsedSec = (Date.now() - spiroStartTime) / 1000.0;
    const timerSpan = document.getElementById('spiro-recording-timer');
    if (timerSpan) timerSpan.textContent = `00:0${Math.min(4, Math.floor(elapsedSec))}`;

    if (elapsedSec >= 4.0) {
      stopSpiroRecording();
    }
  }, 100);
}

function stopSpiroRecording() {
  if (!spiroRecording) return;
  spiroRecording = false;

  if (spiroTimerInterval) clearInterval(spiroTimerInterval);
  if (spiroAnimId) cancelAnimationFrame(spiroAnimId);

  if (spiroMediaStream) {
    spiroMediaStream.getTracks().forEach(track => track.stop());
    spiroMediaStream = null;
  }
  if (spiroAudioContext && spiroAudioContext.state !== 'closed') {
    spiroAudioContext.close();
  }

  const recordBtn = document.getElementById('btn-spiro-record-toggle');
  const recordLabel = document.getElementById('btn-spiro-record-label');
  if (recordBtn) recordBtn.className = 'px-3.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors flex items-center gap-2';
  if (recordLabel) recordLabel.textContent = 'Record Candle Blow (4s)';

  if (spiroPcmSamples.length > 0) {
    const totalLen = spiroPcmSamples.reduce((acc, c) => acc + c.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of spiroPcmSamples) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    const sampleRate = spiroAudioContext ? spiroAudioContext.sampleRate : 16000;
    spiroRecordedBlob = encodePcmWav(merged, sampleRate);

    const playback = document.getElementById('spiro-recorded-playback');
    const analyzeBtn = document.getElementById('btn-analyze-spiro-recorded');
    if (playback && spiroRecordedBlob) {
      playback.src = URL.createObjectURL(spiroRecordedBlob);
      playback.classList.remove('hidden');
    }
    if (analyzeBtn) analyzeBtn.classList.remove('hidden');

    analyzeRecordedSpiroAudio();
  }
}

function drawSpiroOscilloscope() {
  const canvas = document.getElementById('spiro-audio-canvas');
  if (!canvas || !spiroAnalyser) return;

  const ctx = canvas.getContext('2d');
  const bufferLength = spiroAnalyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  const draw = () => {
    if (!spiroRecording) return;
    spiroAnimId = requestAnimationFrame(draw);
    spiroAnalyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = '#070a12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#2dd4bf';
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  };
  draw();
}

async function analyzeRecordedSpiroAudio() {
  if (!spiroRecordedBlob) {
    alert("Please record a candle blow or select a clinical preset.");
    return;
  }

  const analyzeBtn = document.getElementById('btn-analyze-spiro-recorded');
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Processing Flow Dynamics...';
  }

  const formData = new FormData();
  formData.append('file', spiroRecordedBlob, 'forced_exhalation.wav');

  try {
    const res = await fetch('/api/respiratory/test-spirometry', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();
    if (result.status === 'success') {
      renderSpirometryResults(result.data);
    } else {
      alert(result.detail || "Spirometry analysis failed.");
    }
  } catch (err) {
    alert("Spirometry analysis error: " + err.message);
  } finally {
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze Spirometry';
    }
  }
}

async function simulateSpirometryPreset(type) {
  try {
    const res = await fetch(`/api/respiratory/spirometry-preset/${type}`, {
      method: 'POST'
    });
    const json = await res.json();
    if (json.status === 'success') {
      renderSpirometryResults(json.data);
    } else {
      alert(json.detail || "Spirometry preset simulation failed.");
    }
  } catch (err) {
    alert("Spirometry preset error: " + err.message);
  }
}

function renderSpirometryResults(data) {
  const badge = document.getElementById('spiro-risk-badge');
  if (badge) {
    badge.textContent = data.risk_badge;
    if (data.risk_color === 'red') {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-red-950/80 text-red-400 border border-red-800';
    } else if (data.risk_color === 'amber') {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-amber-950/80 text-amber-400 border border-amber-800';
    } else {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800';
    }
  }

  const fev1El = document.getElementById('spiro-metric-fev1');
  if (fev1El) fev1El.textContent = `${data.fev1_liters} L`;

  const fvcEl = document.getElementById('spiro-metric-fvc');
  if (fvcEl) fvcEl.textContent = `${data.fvc_liters} L`;

  const ratioEl = document.getElementById('spiro-metric-ratio');
  if (ratioEl) ratioEl.textContent = `${data.fev1_fvc_percent}%`;

  const pefEl = document.getElementById('spiro-metric-pef');
  if (pefEl) pefEl.textContent = `${data.pef_lps} L/s`;

  const concavityTag = document.getElementById('spiro-concavity-tag');
  if (concavityTag) concavityTag.textContent = `Concavity: ${data.concavity_depth} (${data.concavity_depth > 0.25 ? 'Scooped' : 'Linear'})`;

  const gradeTitle = document.getElementById('spiro-grade-title');
  if (gradeTitle) {
    gradeTitle.textContent = data.simple_status ? `${data.simple_status} (${data.obstruction_grade})` : data.obstruction_grade;
  }

  const interp = document.getElementById('spiro-interpretation');
  if (interp) {
    interp.textContent = data.what_this_means || data.interpretation;
  }

  // Render Flow-Volume Loop Chart (Chart.js)
  renderFlowVolumeChart(data.flow_volume_curve, data.normal_reference_curve);

  // Update multi-modal respiratory state
  state.respResults.spirometry = data;
  evaluateRespiratoryComposite();
}

function renderFlowVolumeChart(patientCurve, normalCurve) {
  const canvas = document.getElementById('spiro-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (spiroChartInstance) {
    spiroChartInstance.destroy();
  }

  spiroChartInstance = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Patient Flow Curve',
          data: patientCurve,
          showLine: true,
          borderColor: '#2dd4bf',
          backgroundColor: 'rgba(45, 212, 191, 0.1)',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: 'Normal Reference Curve',
          data: normalCurve,
          showLine: true,
          borderColor: '#64748b',
          borderWidth: 1.5,
          borderDash: [5, 5],
          pointRadius: 0,
          tension: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: 'Expired Volume (Liters)', color: '#64748b', font: { size: 10 } },
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { color: '#1e293b' }
        },
        y: {
          title: { display: true, text: 'Flow Rate (L/s)', color: '#64748b', font: { size: 10 } },
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { color: '#1e293b' },
          min: 0
        }
      }
    }
  });
}

// =====================================================================
// MODULE 5: CONTACTLESS WEBCAM rPPG & RSA CARDIOPULMONARY COUPLING
// =====================================================================

window.rppgScanning = false;
let rppgMediaStream = null;
let rppgAnimFrameId = null;
let rppgStartTime = null;
let rppgSamplesBuffer = [];
let rppgChartInstance = null;

async function startRppgScan() {
  const video = document.getElementById('rppg-video');
  const idleMsg = document.getElementById('rppg-idle-msg');
  const startBtn = document.getElementById('btn-rppg-start');
  const stopBtn = document.getElementById('btn-rppg-stop');
  const faceGuide = document.getElementById('rppg-face-guide');

  try {
    rppgMediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false
    });
    if (video) video.srcObject = rppgMediaStream;
  } catch (err) {
    alert("Camera permission denied. Please allow webcam access to evaluate optical capillary perfusion.");
    return;
  }

  window.rppgScanning = true;
  rppgStartTime = Date.now();
  rppgSamplesBuffer = [];

  if (idleMsg) idleMsg.classList.add('hidden');
  if (startBtn) startBtn.classList.add('hidden');
  if (stopBtn) stopBtn.classList.remove('hidden');
  if (faceGuide) faceGuide.classList.add('tracking');

  const pulseDot = document.getElementById('rppg-pulse-dot');
  if (pulseDot) pulseDot.className = 'h-2 w-2 rounded-full bg-emerald-400 pulse-indicator';

  // Off-screen canvas for pixel sampling
  const hiddenCanvas = document.getElementById('rppg-hidden-canvas') || document.createElement('canvas');
  hiddenCanvas.width = 64;
  hiddenCanvas.height = 48;
  const hCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

  const scanDurationMs = 15000; // 15 seconds

  const processFrame = () => {
    if (!window.rppgScanning) return;

    const elapsed = Date.now() - rppgStartTime;
    const progress = Math.min(100, Math.round((elapsed / scanDurationMs) * 100));

    const pBar = document.getElementById('rppg-progress-bar');
    const pText = document.getElementById('rppg-progress-text');
    const timer = document.getElementById('rppg-timer');
    if (pBar) pBar.style.width = `${progress}%`;
    if (pText) pText.textContent = `${progress}%`;
    if (timer) timer.textContent = `00:${String(Math.floor(elapsed / 1000)).padStart(2, '0')}`;

    // Sample Green channel ROI (forehead region: center-top 30% of frame)
    if (video && video.readyState >= 2) {
      hCtx.drawImage(video, 0, 0, 64, 48);
      // Sample center rectangle (x: 20..44, y: 10..26)
      const frameData = hCtx.getImageData(20, 10, 24, 16).data;
      let sumGreen = 0;
      const count = frameData.length / 4;
      for (let i = 0; i < frameData.length; i += 4) {
        sumGreen += frameData[i + 1]; // Green channel
      }
      const avgGreen = sumGreen / count;
      rppgSamplesBuffer.push({ t: elapsed, val: avgGreen });
    }

    if (elapsed >= scanDurationMs) {
      finishRppgScan();
    } else {
      rppgAnimFrameId = requestAnimationFrame(processFrame);
    }
  };

  rppgAnimFrameId = requestAnimationFrame(processFrame);
}

function stopRppgScan() {
  window.rppgScanning = false;
  if (rppgAnimFrameId) cancelAnimationFrame(rppgAnimFrameId);

  if (rppgMediaStream) {
    rppgMediaStream.getTracks().forEach(track => track.stop());
    rppgMediaStream = null;
  }

  const startBtn = document.getElementById('btn-rppg-start');
  const stopBtn = document.getElementById('btn-rppg-stop');
  const faceGuide = document.getElementById('rppg-face-guide');
  const pulseDot = document.getElementById('rppg-pulse-dot');

  if (startBtn) startBtn.classList.remove('hidden');
  if (stopBtn) stopBtn.classList.add('hidden');
  if (faceGuide) faceGuide.classList.remove('tracking');
  if (pulseDot) pulseDot.className = 'h-2 w-2 rounded-full bg-slate-600';
}

async function finishRppgScan() {
  stopRppgScan();

  if (rppgSamplesBuffer.length < 90) {
    alert("Insufficient optical scan frames captured. Please ensure steady lighting.");
    return;
  }

  const timer = document.getElementById('rppg-timer');
  if (timer) timer.textContent = 'Analyzing...';

  try {
    const res = await fetch('/api/respiratory/test-rppg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: rppgSamplesBuffer })
    });

    const result = await res.json();
    if (result.status === 'success') {
      renderRppgResults(result.data);
    } else {
      alert(result.detail || "rPPG telemetry processing failed.");
    }
  } catch (err) {
    alert("rPPG processing error: " + err.message);
  } finally {
    if (timer) timer.textContent = '00:15';
  }
}

async function simulateRppgPreset(type) {
  // Generate 15.0 seconds of synthetic green channel capillary timeseries at 30 fps (450 frames)
  const fs = 30.0;
  const dur = 15.0;
  const numPts = Math.floor(fs * dur);
  const syntheticSamples = [];

  for (let i = 0; i < numPts; i++) {
    const t = i / fs;
    let val;
    if (type === 'healthy') {
      // 72 BPM cardiac pulse coupled with 15 BPM respiration (strong RSA modulation)
      const respFreq = 15.0 / 60.0; // 0.25 Hz
      const respWave = Math.sin(2 * Math.PI * respFreq * t);
      // Instantaneous heart rate modulates with resp wave (62 to 82 BPM)
      const instHr = 72.0 + 12.0 * respWave;
      const cardiacWave = Math.sin(2 * Math.PI * (instHr / 60.0) * t);
      val = 140.0 + 4.5 * cardiacWave + 2.8 * respWave + (Math.random() - 0.5) * 0.4;
    } else {
      // Decoupled / Blunted RSA: 82 BPM pulse monotone with no respiratory rhythm response
      const cardiacWave = Math.sin(2 * Math.PI * (82.0 / 60.0) * t);
      const respWave = 0.5 * Math.sin(2 * Math.PI * 0.35 * t); // weak irregular breath
      val = 138.0 + 4.2 * cardiacWave + respWave + (Math.random() - 0.5) * 0.6;
    }
    syntheticSamples.push({ t: Math.round(t * 1000), val: Math.round(val * 100) / 100 });
  }

  try {
    const res = await fetch('/api/respiratory/test-rppg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: syntheticSamples })
    });

    const result = await res.json();
    if (result.status === 'success') {
      renderRppgResults(result.data);
    }
  } catch (err) {
    alert("rPPG simulation error: " + err.message);
  }
}

function renderRppgResults(data) {
  const badge = document.getElementById('rppg-risk-badge');
  if (badge) {
    badge.textContent = data.risk_badge;
    if (data.risk_color === 'red') {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-red-950/80 text-red-400 border border-red-800';
    } else if (data.risk_color === 'amber') {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-amber-950/80 text-amber-400 border border-amber-800';
    } else {
      badge.className = 'px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800';
    }
  }

  const hrEl = document.getElementById('rppg-metric-hr');
  if (hrEl) hrEl.textContent = `${data.heart_rate_bpm} BPM`;

  const rrEl = document.getElementById('rppg-metric-rr');
  if (rrEl) rrEl.textContent = `${data.respiratory_rate_bpm} BPM`;

  const eiEl = document.getElementById('rppg-metric-ei');
  if (eiEl) eiEl.textContent = data.ei_ratio;

  const rmssdEl = document.getElementById('rppg-metric-rmssd');
  if (rmssdEl) rmssdEl.textContent = `${data.hrv_rmssd_ms} ms`;

  const coherenceTag = document.getElementById('rppg-coherence-tag');
  if (coherenceTag) coherenceTag.textContent = `Coherence: ${data.coherence_score} (${data.autonomic_status})`;

  const statusTitle = document.getElementById('rppg-status-title');
  if (statusTitle) statusTitle.textContent = data.autonomic_status;

  const interp = document.getElementById('rppg-interpretation');
  if (interp) interp.textContent = data.interpretation;

  // Render Chart.js Synchronous Dual Waveforms
  if (data.waveforms) {
    renderRppgChart(data.waveforms.pulse, data.waveforms.respiration);
  }

  // Update multi-modal respiratory state
  state.respResults.vitals = data;
  evaluateRespiratoryComposite();
}

function renderRppgChart(pulseData, respData) {
  const canvas = document.getElementById('rppg-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (rppgChartInstance) {
    rppgChartInstance.destroy();
  }

  rppgChartInstance = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Cardiac Pulse (rPPG)',
          data: pulseData,
          showLine: true,
          borderColor: '#f87171',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3
        },
        {
          label: 'Respiration Rhythm',
          data: respData,
          showLine: true,
          borderColor: '#38bdf8',
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: 'Time (seconds)', color: '#64748b', font: { size: 10 } },
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { color: '#1e293b' }
        },
        y: {
          display: false,
          grid: { display: false }
        }
      }
    }
  });
}

// =====================================================================
// MODULE NAVIGATION & INTERACTIVE GUIDES
// =====================================================================

function toggleGuide(guideId) {
  const guide = document.getElementById(guideId);
  const icon = document.getElementById(`${guideId}-icon`);
  if (!guide) return;
  const isHidden = guide.classList.toggle('hidden');
  if (icon) {
    icon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';
  }
}

function filterRespStation(station) {
  // Normalize alias
  if (station === 'steth') station = 'auscultation';
  if (station === 'spiro') station = 'spirometry';

  const pillMap = {
    'all': 'pill-resp-all',
    'auscultation': 'pill-resp-ausc',
    'spirometry': 'pill-resp-spiro',
    'rppg': 'pill-resp-rppg'
  };

  Object.entries(pillMap).forEach(([stKey, pillId]) => {
    const pill = document.getElementById(pillId);
    if (pill) {
      if (stKey === station) {
        pill.className = 'px-3 py-1 rounded font-medium bg-teal-900/60 text-teal-300 border border-teal-700 transition-colors';
      } else {
        pill.className = 'px-3 py-1 rounded font-medium text-slate-400 hover:text-slate-200 transition-colors';
      }
    }
  });

  const ausc = document.getElementById('station-resp-auscultation') || document.getElementById('station-resp-steth');
  const spiro = document.getElementById('station-resp-spirometry') || document.getElementById('station-resp-spiro');
  const rppg = document.getElementById('station-resp-rppg');

  if (station === 'all') {
    if (ausc) ausc.classList.remove('hidden');
    if (spiro) spiro.classList.remove('hidden');
    if (rppg) rppg.classList.remove('hidden');
  } else if (station === 'auscultation') {
    if (ausc) ausc.classList.remove('hidden');
    if (spiro) spiro.classList.add('hidden');
    if (rppg) rppg.classList.add('hidden');
  } else if (station === 'spirometry') {
    if (ausc) ausc.classList.add('hidden');
    if (spiro) spiro.classList.remove('hidden');
    if (rppg) rppg.classList.add('hidden');
  } else if (station === 'rppg') {
    if (ausc) ausc.classList.add('hidden');
    if (spiro) spiro.classList.add('hidden');
    if (rppg) rppg.classList.remove('hidden');
  }
}

function filterParkStation(station) {
  // Normalize alias
  if (station === 'vocal') station = 'phonation';
  if (station === 'calib') station = 'calibration';

  const pillMap = {
    'all': 'pill-park-all',
    'phonation': 'pill-park-phon',
    'spiral': 'pill-park-spiral',
    'calibration': 'pill-park-calib'
  };

  Object.entries(pillMap).forEach(([stKey, pillId]) => {
    const pill = document.getElementById(pillId);
    if (pill) {
      if (stKey === station) {
        pill.className = 'px-3 py-1 rounded font-medium bg-amber-950/60 text-amber-300 border border-amber-800 transition-colors';
      } else {
        pill.className = 'px-3 py-1 rounded font-medium text-slate-400 hover:text-slate-200 transition-colors';
      }
    }
  });

  const phon = document.getElementById('station-park-phonation') || document.getElementById('station-park-vocal');
  const spiral = document.getElementById('station-park-spiral');
  const calib = document.getElementById('station-park-calibration');

  if (station === 'all') {
    if (phon) phon.classList.remove('hidden');
    if (spiral) spiral.classList.remove('hidden');
    if (calib) calib.classList.remove('hidden');
  } else if (station === 'phonation') {
    if (phon) phon.classList.remove('hidden');
    if (spiral) spiral.classList.add('hidden');
    if (calib) calib.classList.add('hidden');
  } else if (station === 'spiral') {
    if (phon) phon.classList.add('hidden');
    if (spiral) spiral.classList.remove('hidden');
    if (calib) calib.classList.add('hidden');
    setTimeout(initSpiralCanvas, 50);
  } else if (station === 'calibration') {
    if (phon) phon.classList.add('hidden');
    if (spiral) spiral.classList.add('hidden');
    if (calib) calib.classList.remove('hidden');
  }
}

// =====================================================================
// MULTI-MODAL COMPOSITE DIAGNOSTICS & EVALUATION
// =====================================================================

async function evaluateRespiratoryComposite() {
  if (!state.respResults.auscultation && !state.respResults.spirometry && !state.respResults.vitals) {
    return;
  }

  try {
    const res = await fetch('/api/respiratory/composite-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        auscultation: state.respResults.auscultation,
        spirometry: state.respResults.spirometry,
        vitals: state.respResults.vitals
      })
    });
    const json = await res.json();
    if (json.status === 'success') {
      const data = json.data || json;
      state.respComposite = data;
      renderRespiratoryCompositeUI(data);
    }
  } catch (err) {
    console.warn("Respiratory composite prediction failed:", err);
  }
}

function renderRespiratoryCompositeUI(data) {
  const badge = document.getElementById('resp-composite-badge');
  const summary = document.getElementById('resp-composite-summary');
  const findings = document.getElementById('resp-composite-findings');

  const riskBadge = data.risk_badge || data.overall_badge || data.composite_risk_level || 'Awaiting Test Data';
  if (badge) {
    badge.textContent = riskBadge;
    if (data.risk_color === 'red') {
      badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-red-950 text-red-300 border border-red-800';
    } else if (data.risk_color === 'amber') {
      badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-amber-950 text-amber-300 border border-amber-800';
    } else {
      badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800';
    }
  }

  if (summary) {
    summary.textContent = data.plain_language_summary || data.plain_summary || 'All completed tests evaluated.';
  }

  const items = data.findings || data.individual_findings;
  if (findings && items && items.length > 0) {
    findings.classList.remove('hidden');
    findings.innerHTML = items.map(f => {
      if (typeof f === 'string') {
        return `<div class="flex items-start gap-2 py-1.5 border-t border-slate-800/80">
          <span class="h-2 w-2 rounded-full bg-teal-400 mt-1 shrink-0"></span>
          <span class="text-xs text-slate-200">${f}</span>
        </div>`;
      } else {
        return `<div class="flex items-start gap-2 py-1.5 border-t border-slate-800/80">
          <span class="font-bold text-teal-400 shrink-0 text-xs">${f.station || f.test || 'Finding'}:</span>
          <span class="text-xs text-slate-200">${f.finding} ${f.clinical_meaning ? `<span class="text-slate-400">(${f.clinical_meaning})</span>` : ''}</span>
        </div>`;
      }
    }).join('');
  }
}

async function evaluateParkinsonComposite() {
  if (!state.parkResults.vocal && !state.parkResults.spiral) {
    return;
  }

  try {
    const res = await fetch('/api/parkinson/composite-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vocal: state.parkResults.vocal,
        spiral: state.parkResults.spiral
      })
    });
    const json = await res.json();
    if (json.status === 'success') {
      const data = json.data || json;
      state.parkComposite = data;
      renderParkinsonCompositeUI(data);
    }
  } catch (err) {
    console.warn("Parkinson composite prediction failed:", err);
  }
}

function renderParkinsonCompositeUI(data) {
  const badge = document.getElementById('park-composite-badge');
  const summary = document.getElementById('park-composite-summary');
  const findings = document.getElementById('park-composite-findings');

  const riskBadge = data.risk_badge || data.overall_badge || data.composite_risk_level || 'Awaiting Test Data';
  if (badge) {
    badge.textContent = riskBadge;
    if (data.risk_color === 'red') {
      badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-red-950 text-red-300 border border-red-800';
    } else if (data.risk_color === 'amber') {
      badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-amber-950 text-amber-300 border border-amber-800';
    } else {
      badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800';
    }
  }

  if (summary) {
    summary.textContent = data.plain_language_summary || data.plain_summary || 'All completed tests evaluated.';
  }

  const items = data.findings || data.individual_findings;
  if (findings && items && items.length > 0) {
    findings.classList.remove('hidden');
    findings.innerHTML = items.map(f => {
      if (typeof f === 'string') {
        return `<div class="flex items-start gap-2 py-1.5 border-t border-slate-800/80">
          <span class="h-2 w-2 rounded-full bg-amber-400 mt-1 shrink-0"></span>
          <span class="text-xs text-slate-200">${f}</span>
        </div>`;
      } else {
        return `<div class="flex items-start gap-2 py-1.5 border-t border-slate-800/80">
          <span class="font-bold text-amber-400 shrink-0 text-xs">${f.test || f.station || 'Finding'}:</span>
          <span class="text-xs text-slate-200">${f.finding} ${f.clinical_meaning ? `<span class="text-slate-400">(${f.clinical_meaning})</span>` : ''}</span>
        </div>`;
      }
    }).join('');
  }
}

// =====================================================================
// CLINICAL DOCTOR REPORT GENERATOR (PRINT & PDF)
// =====================================================================

function openDoctorReport(centerType) {
  const container = document.getElementById('report-paper-content');
  const modal = document.getElementById('report-modal');
  if (!container || !modal) return;

  const dateStr = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const reportId = `NB-${centerType === 'respiratory' ? 'RESP' : 'PARK'}-${Math.floor(100000 + Math.random() * 900000)}`;

  let html = '';

  if (centerType === 'respiratory') {
    const comp = state.respComposite || {
      overall_status: 'Awaiting Completed Station',
      composite_risk_level: 'Screening Incomplete',
      risk_badge: 'Awaiting Completed Station',
      risk_color: 'slate',
      confidence_score: 0.85,
      plain_language_summary: 'Partial respiratory tests recorded. Recommend completing auscultation, spirometry, and vitals.',
      plain_summary: 'Partial respiratory tests recorded. Recommend completing auscultation, spirometry, and vitals.',
      findings: []
    };
    const ausc = state.respResults.auscultation;
    const spiro = state.respResults.spirometry;
    const rppg = state.respResults.vitals;

    const riskColorClass = comp.risk_color === 'red' ? 'text-red-700 bg-red-50 border-red-300' :
      (comp.risk_color === 'amber' ? 'text-amber-700 bg-amber-50 border-amber-300' : 'text-emerald-700 bg-emerald-50 border-emerald-300');

    html = `
      <div class="border-b-2 border-slate-900 pb-4 flex flex-col sm:flex-row justify-between items-start gap-2">
        <div>
          <h1 class="text-xl font-black tracking-tight text-slate-900">NEUROBREATHE CLINICAL DECISION SUPPORT SYSTEM</h1>
          <p class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Multi-Modal Respiratory & Cardiopulmonary Health Evaluation</p>
          <p class="text-[11px] text-slate-500 mt-0.5">Division of Pulmonary Medicine & Digital Acoustic Biomarkers</p>
        </div>
        <div class="text-left sm:text-right text-xs text-slate-600 font-mono">
          <p class="font-bold text-slate-900">REPORT ID: ${reportId}</p>
          <p>Generated: ${dateStr}</p>
          <p>${timeStr}</p>
        </div>
      </div>

      <!-- Editable Patient & Consultation Details -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-50 p-3.5 rounded border border-slate-200 text-xs">
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Patient Name</label>
          <input type="text" id="report-patient-name" value="Patient Screening Consultation" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-semibold focus:outline-none focus:border-teal-500">
        </div>
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Patient ID / MRN</label>
          <input type="text" id="report-patient-mrn" value="PT-${Math.floor(10000 + Math.random() * 90000)}" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-mono font-semibold focus:outline-none focus:border-teal-500">
        </div>
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Age / Biological Sex</label>
          <input type="text" id="report-patient-age-sex" value="62 / Male" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-semibold focus:outline-none focus:border-teal-500">
        </div>
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Referring Clinic / Unit</label>
          <input type="text" id="report-patient-clinic" value="NeuroBreathe Telehealth Clinic" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-semibold focus:outline-none focus:border-teal-500">
        </div>
      </div>

      <!-- Primary Diagnostic Impression -->
      <div class="p-4 rounded-lg border ${riskColorClass} space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold uppercase tracking-wide">Overall Respiratory Diagnosis:</span>
          <span class="px-2.5 py-0.5 rounded text-xs font-bold uppercase border">${comp.risk_badge || comp.overall_badge || comp.composite_risk_level}</span>
        </div>
        <h2 class="text-base font-bold text-slate-900">${comp.overall_status || comp.composite_risk_level}</h2>
        <div class="text-xs text-slate-800 space-y-1">
          <p class="font-medium leading-relaxed">${comp.plain_language_summary || comp.plain_summary}</p>
        </div>
        <div class="text-[11px] text-slate-600 pt-1 border-t border-slate-300">
          <strong>Physician Synopsis:</strong> Evaluated via tri-modal diagnostic framework combining stethoscope lung acoustic auscultation (60 spectral features), virtual dynamic acoustic spirometry (FEV1/FVC flow mechanics), and optical rPPG cardiopulmonary coupling.
        </div>
      </div>

      <!-- Diagnostic Station Biomarkers Table -->
      <div class="space-y-2">
        <h3 class="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-200 pb-1">Station Telemetry & Objective Clinical Metrics</h3>
        <table class="w-full text-xs text-left border-collapse border border-slate-200">
          <thead>
            <tr class="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
              <th class="p-2 border-r border-slate-200">Diagnostic Station</th>
              <th class="p-2 border-r border-slate-200">Key Parameters Measured</th>
              <th class="p-2 border-r border-slate-200">Observed Telemetry</th>
              <th class="p-2 border-r border-slate-200">Reference Norm</th>
              <th class="p-2 text-right">Clinical Stratum</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-200 text-slate-800">
            <tr>
              <td class="p-2 font-medium border-r border-slate-200">Station 1: Lung Sounds</td>
              <td class="p-2 border-r border-slate-200">Acoustic Wheeze / Crackle Classification</td>
              <td class="p-2 font-mono border-r border-slate-200">${ausc ? (ausc.prediction + ' (' + Math.round(ausc.confidence * 100) + '% confidence)') : 'Not tested'}</td>
              <td class="p-2 border-r border-slate-200">Vesicular Normal</td>
              <td class="p-2 text-right font-semibold ${ausc && ausc.prediction === 'COPD' ? 'text-red-600' : 'text-emerald-600'}">${ausc ? (ausc.risk_badge || ausc.prediction) : '--'}</td>
            </tr>
            <tr>
              <td class="p-2 font-medium border-r border-slate-200">Station 2: Spirometry</td>
              <td class="p-2 border-r border-slate-200">FEV1 / FVC Ratio & Peak Flow (PEF)</td>
              <td class="p-2 font-mono border-r border-slate-200">${spiro ? ('Ratio: ' + spiro.fev1_fvc_percent + '% | FEV1: ' + spiro.fev1_liters + 'L | PEF: ' + spiro.pef_lps + ' L/s') : 'Not tested'}</td>
              <td class="p-2 border-r border-slate-200">Ratio &ge; 70.0%</td>
              <td class="p-2 text-right font-semibold ${spiro && spiro.fev1_fvc_percent < 70 ? 'text-red-600' : 'text-emerald-600'}">${spiro ? (spiro.simple_status || spiro.obstruction_grade) : '--'}</td>
            </tr>
            <tr>
              <td class="p-2 font-medium border-r border-slate-200">Station 3: Webcam Vitals</td>
              <td class="p-2 border-r border-slate-200">RSA Expiratory:Inspiratory Coupling & BPM</td>
              <td class="p-2 font-mono border-r border-slate-200">${rppg ? ('HR: ' + rppg.heart_rate_bpm + ' BPM | RR: ' + rppg.respiratory_rate_bpm + ' BPM | RSA: ' + rppg.ei_ratio) : 'Not tested'}</td>
              <td class="p-2 border-r border-slate-200">RSA E:I &ge; 1.20 | RR: 12-20</td>
              <td class="p-2 text-right font-semibold ${rppg && rppg.ei_ratio < 1.20 ? 'text-amber-600' : 'text-emerald-600'}">${rppg ? rppg.autonomic_status : '--'}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Evidence-Based Care Tips & Post-Prediction Guidance -->
      <div class="space-y-2 bg-slate-50 p-3 rounded border border-slate-200 text-xs text-slate-800">
        <h4 class="font-bold text-slate-900 uppercase">Recommended Patient Care Tips & Exercise Plan:</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-teal-700 block">Pursed-Lip Breathing Technique</strong>
            Inhale slowly through your nose for 2 counts. Pucker your lips like blowing out candles, and exhale gently for 4 counts to release trapped stale air.
          </div>
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-teal-700 block">Diaphragmatic (Belly) Breathing</strong>
            Sit upright with one hand on your abdomen. Inhale through the nose allowing belly to expand outward, then exhale relaxed as belly falls.
          </div>
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-teal-700 block">Hydration & Mucus Thinning</strong>
            Drink 6-8 glasses of warm water daily to keep bronchial secretions thin and easy to clear naturally. Avoid smoke and strong aerosols.
          </div>
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-red-700 block">Urgent Red-Flag Warnings</strong>
            Seek immediate medical attention if experiencing sudden severe dyspnea, blueness around lips/fingers, or high fever with purulent sputum.
          </div>
        </div>
      </div>

      <!-- Clinician Notes (Editable) -->
      <div class="space-y-1 bg-slate-50 p-3 rounded border border-slate-200 text-xs text-slate-800">
        <label class="font-bold text-slate-900 uppercase block">Physician / Clinician Assessment Notes & Orders:</label>
        <textarea id="report-physician-notes" rows="3" class="w-full bg-white border border-slate-300 rounded p-2 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-teal-600" placeholder="Type doctor clinical remarks, prescribed bronchodilators, pulmonary rehab referrals, or follow-up schedule here..."></textarea>
      </div>

      <!-- Physician Attestation & Signature Line -->
      <div class="pt-4 border-t border-slate-300 grid grid-cols-2 gap-6 text-xs text-slate-600">
        <div>
          <p class="font-semibold text-slate-900">Reviewing Pulmonologist / Physician Attestation:</p>
          <p class="text-[11px] text-slate-500 mt-0.5">I certify that I have reviewed this automated multi-modal screening record.</p>
          <div class="mt-7 border-b border-slate-400 w-4/5"></div>
          <p class="text-[10px] text-slate-400 mt-1">Physician Signature & License Number / NPI</p>
        </div>
        <div class="text-right">
          <p class="font-semibold text-slate-900">Attestation Date & Time:</p>
          <p class="text-[11px] text-slate-500 mt-0.5">${dateStr} - ${timeStr}</p>
          <div class="mt-7 border-b border-slate-400 ml-auto w-3/5"></div>
          <p class="text-[10px] text-slate-400 mt-1">Authorized Medical Stamp / Signature</p>
        </div>
      </div>

      <div class="text-[10px] text-slate-500 border-t border-slate-200 pt-2 text-center">
        * Clinical Decision Support Notice: NeuroBreathe AI is an adjunctive clinical decision support screening tool. Final clinical diagnosis must be confirmed via formal laboratory spirometry (PFT) or stethoscope examination by a licensed medical physician.
      </div>
    `;
  } else {
    // Parkinson Report
    const comp = state.parkComposite || {
      overall_status: 'Awaiting Completed Station',
      composite_risk_level: 'Screening Incomplete',
      risk_badge: 'Awaiting Completed Station',
      risk_color: 'slate',
      confidence_score: 0.85,
      plain_language_summary: 'Partial Parkinson neuromotor tests recorded. Recommend completing sustained phonation (MPT) and spiral drawing.',
      plain_summary: 'Partial Parkinson neuromotor tests recorded. Recommend completing sustained phonation (MPT) and spiral drawing.',
      findings: []
    };
    const voc = state.parkResults.vocal;
    const spir = state.parkResults.spiral;

    const riskColorClass = comp.risk_color === 'red' ? 'text-red-700 bg-red-50 border-red-300' :
      (comp.risk_color === 'amber' ? 'text-amber-700 bg-amber-50 border-amber-300' : 'text-emerald-700 bg-emerald-50 border-emerald-300');

    html = `
      <div class="border-b-2 border-slate-900 pb-4 flex flex-col sm:flex-row justify-between items-start gap-2">
        <div>
          <h1 class="text-xl font-black tracking-tight text-slate-900">NEUROBREATHE CLINICAL DECISION SUPPORT SYSTEM</h1>
          <p class="text-xs font-semibold text-slate-600 uppercase tracking-wider">Multi-Modal Parkinson Neurological & Motor Screening Report</p>
          <p class="text-[11px] text-slate-500 mt-0.5">Division of Movement Disorders & Acoustic Neurology</p>
        </div>
        <div class="text-left sm:text-right text-xs text-slate-600 font-mono">
          <p class="font-bold text-slate-900">REPORT ID: ${reportId}</p>
          <p>Generated: ${dateStr}</p>
          <p>${timeStr}</p>
        </div>
      </div>

      <!-- Editable Patient & Consultation Details -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-50 p-3.5 rounded border border-slate-200 text-xs">
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Patient Name</label>
          <input type="text" id="report-patient-name" value="Patient Screening Consultation" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-semibold focus:outline-none focus:border-amber-500">
        </div>
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Patient ID / MRN</label>
          <input type="text" id="report-patient-mrn" value="PT-${Math.floor(10000 + Math.random() * 90000)}" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-mono font-semibold focus:outline-none focus:border-amber-500">
        </div>
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Age / Biological Sex</label>
          <input type="text" id="report-patient-age-sex" value="68 / Male" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-semibold focus:outline-none focus:border-amber-500">
        </div>
        <div>
          <label class="text-slate-500 block text-[10px] uppercase font-semibold">Referring Clinic / Unit</label>
          <input type="text" id="report-patient-clinic" value="NeuroBreathe Telehealth Clinic" class="w-full bg-white border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-900 font-semibold focus:outline-none focus:border-amber-500">
        </div>
      </div>

      <!-- Primary Diagnostic Impression -->
      <div class="p-4 rounded-lg border ${riskColorClass} space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold uppercase tracking-wide">Overall Neurological Diagnosis:</span>
          <span class="px-2.5 py-0.5 rounded text-xs font-bold uppercase border">${comp.risk_badge || comp.overall_badge || comp.composite_risk_level}</span>
        </div>
        <h2 class="text-base font-bold text-slate-900">${comp.overall_status || comp.composite_risk_level}</h2>
        <div class="text-xs text-slate-800 space-y-1">
          <p class="font-medium leading-relaxed">${comp.plain_language_summary || comp.plain_summary}</p>
        </div>
        <div class="text-[11px] text-slate-600 pt-1 border-t border-slate-300">
          <strong>Physician Synopsis:</strong> Dual-stream predictive modeling evaluating vocal acoustic perturbation (MDVP jitter, shimmer, HNR, MPT stamina) paired with continuous digital Archimedes spiral drawing kinematics (jerk smoothness, micrographia decay, 4-7 Hz action tremor).
        </div>
      </div>

      <!-- Diagnostic Station Biomarkers Table -->
      <div class="space-y-2">
        <h3 class="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-200 pb-1">Objective Neuromotor & Vocal Telemetry</h3>
        <table class="w-full text-xs text-left border-collapse border border-slate-200">
          <thead>
            <tr class="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
              <th class="p-2 border-r border-slate-200">Diagnostic Station</th>
              <th class="p-2 border-r border-slate-200">Biomarker / Kinematic Parameter</th>
              <th class="p-2 border-r border-slate-200">Observed Telemetry</th>
              <th class="p-2 border-r border-slate-200">Normal Cutoff</th>
              <th class="p-2 text-right">Clinical Stratum</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-200 text-slate-800">
            <tr>
              <td class="p-2 font-medium border-r border-slate-200">Station 1: Voice Phonation</td>
              <td class="p-2 border-r border-slate-200">Dysphonia Probability (XGBoost 22-Feat)</td>
              <td class="p-2 font-mono border-r border-slate-200">${voc ? ((voc.probability_parkinson * 100).toFixed(1) + '% PD Probability (' + (voc.diagnosis || 'Evaluated') + ')') : 'Not tested'}</td>
              <td class="p-2 border-r border-slate-200">&lt; 50.0% Healthy</td>
              <td class="p-2 text-right font-semibold ${voc && (voc.status === 1 || voc.risk_color === 'red') ? 'text-red-600' : 'text-emerald-600'}">${voc ? (voc.risk_badge || voc.risk_level) : '--'}</td>
            </tr>
            <tr>
              <td class="p-2 font-medium border-r border-slate-200">Station 1: MPT Stamina</td>
              <td class="p-2 border-r border-slate-200">Maximum Phonation Hold Duration</td>
              <td class="p-2 font-mono border-r border-slate-200">${voc && voc.mpt ? (voc.mpt.duration_seconds + ' seconds') : (voc ? 'Biomarker Preset (Fo: ' + (voc.features ? voc.features['MDVP:Fo(Hz)'] : '197') + ' Hz)' : 'Not tested')}</td>
              <td class="p-2 border-r border-slate-200">&ge; 10.0 seconds</td>
              <td class="p-2 text-right font-semibold ${voc && voc.mpt && voc.mpt.is_reduced ? 'text-amber-600' : 'text-emerald-600'}">${voc && voc.mpt ? voc.mpt.status : 'Normal'}</td>
            </tr>
            <tr>
              <td class="p-2 font-medium border-r border-slate-200">Station 2: Spiral Drawing</td>
              <td class="p-2 border-r border-slate-200">Kinematic Jerk Smoothness</td>
              <td class="p-2 font-mono border-r border-slate-200">${spir && spir.metrics ? (spir.metrics.drawing_jerk + ' | Duration: ' + spir.metrics.duration_seconds + 's') : 'Not tested'}</td>
              <td class="p-2 border-r border-slate-200">&lt; 25.0 Smooth</td>
              <td class="p-2 text-right font-semibold ${spir && spir.risk_color === 'red' ? 'text-red-600' : 'text-emerald-600'}">${spir && spir.metrics ? (spir.metrics.jerk_status || spir.risk_level) : '--'}</td>
            </tr>
            <tr>
              <td class="p-2 font-medium border-r border-slate-200">Station 2: Tremor Peak</td>
              <td class="p-2 border-r border-slate-200">4-7 Hz Kinematic Tremor Power & Decay</td>
              <td class="p-2 font-mono border-r border-slate-200">${spir && spir.metrics ? ('Peak: ' + spir.metrics.tremor_peak_power + ' | Decay: ' + spir.metrics.micrographia_decay_pct + '%') : 'Not tested'}</td>
              <td class="p-2 border-r border-slate-200">Peak &lt; 0.15 | Decay &lt; 15%</td>
              <td class="p-2 text-right font-semibold ${spir && spir.metrics && spir.metrics.micrographia_decay_pct > 20 ? 'text-amber-600' : 'text-emerald-600'}">${spir && spir.metrics ? spir.metrics.micrographia_status : '--'}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Evidence-Based Care Tips & Post-Prediction Guidance -->
      <div class="space-y-2 bg-slate-50 p-3 rounded border border-slate-200 text-xs text-slate-800">
        <h4 class="font-bold text-slate-900 uppercase">Recommended Patient Care Tips & Exercise Plan:</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-amber-700 block">Sustained "Ahhh" Loudness Drills</strong>
            Vocalize a loud, clear "aaah" at comfortable pitch, holding steady for 10-15 seconds. Repeat 5 times daily to preserve vocal fold strength and lung-larynx support.
          </div>
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-amber-700 block">Finger Tapping & Hand Opening</strong>
            Open and close hands widely and quickly for 10 repetitions, then tap each finger to your thumb in sequence to maintain fine motor dexterity and ease morning stiffness.
          </div>
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-amber-700 block">Big & Bold Handwriting Practice</strong>
            Spend 5 minutes daily writing on lined paper with deliberately large, exaggerated letters to re-calibrate brain motor feedback and combat micrographia.
          </div>
          <div class="p-2 rounded bg-white border border-slate-200">
            <strong class="text-red-700 block">When to Consult a Neurologist</strong>
            Schedule specialist evaluation if noticing resting tremors in hands or chin, unexplained muscle stiffness, slowness getting up from chairs, or marked voice softening.
          </div>
        </div>
      </div>

      <!-- Clinician Notes (Editable) -->
      <div class="space-y-1 bg-slate-50 p-3 rounded border border-slate-200 text-xs text-slate-800">
        <label class="font-bold text-slate-900 uppercase block">Neurologist / Clinician Assessment Notes & Orders:</label>
        <textarea id="report-physician-notes" rows="3" class="w-full bg-white border border-slate-300 rounded p-2 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-amber-600" placeholder="Type doctor neurological observations, MDS-UPDRS Part III staging notes, or medication adjustment recommendations here..."></textarea>
      </div>

      <!-- Physician Attestation & Signature Line -->
      <div class="pt-4 border-t border-slate-300 grid grid-cols-2 gap-6 text-xs text-slate-600">
        <div>
          <p class="font-semibold text-slate-900">Reviewing Neurologist Attestation:</p>
          <p class="text-[11px] text-slate-500 mt-0.5">I certify that I have reviewed this automated multi-modal screening record.</p>
          <div class="mt-7 border-b border-slate-400 w-4/5"></div>
          <p class="text-[10px] text-slate-400 mt-1">Physician Signature & License Number / NPI</p>
        </div>
        <div class="text-right">
          <p class="font-semibold text-slate-900">Attestation Date & Time:</p>
          <p class="text-[11px] text-slate-500 mt-0.5">${dateStr} - ${timeStr}</p>
          <div class="mt-7 border-b border-slate-400 ml-auto w-3/5"></div>
          <p class="text-[10px] text-slate-400 mt-1">Authorized Medical Stamp / Signature</p>
        </div>
      </div>

      <div class="text-[10px] text-slate-500 border-t border-slate-200 pt-2 text-center">
        * Clinical Decision Support Notice: NeuroBreathe AI is an adjunctive screening tool. It does not replace a comprehensive neurological examination by a board-certified neurologist.
      </div>
    `;
  }

  container.innerHTML = html;
  modal.classList.remove('hidden');
}

function closeDoctorReport() {
  const modal = document.getElementById('report-modal');
  if (modal) modal.classList.add('hidden');
}

function printDoctorReport() {
  window.print();
}

// Function aliases for compatibility
const analyzeSpiralPoints = analyzeSpiralDrawing;
const simulateSpiralPreset = loadPresetSpiral;
