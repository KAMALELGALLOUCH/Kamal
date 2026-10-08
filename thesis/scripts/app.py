"""
Multimodal Emotion Recognition — Prototype Tool
================================================
Flask API + Web Interface with MELD + IEMOCAP support

Endpoints:
  GET  /              → Web UI
  POST /predict       → JSON: {model, text, audio_base64} → emotion prediction
  GET  /health        → API health check
  
"""

import os
import sys
import io
import base64
import logging
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModel

sys.path.append('/data8/luoyan/Kamal/thesis')

DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'
MELD_CKPT       = '/data8/luoyan/Kamal/thesis/outputs/best_multimodal_model.pt'
IEMOCAP_CKPT    = '/data8/luoyan/Kamal/thesis/outputs/iemocap/best_iemocap_model.pt'
MAX_LEN         = 128
PORT            = 5000

MELD_ID2EMOTION = {
    0: 'neutral', 1: 'joy', 2: 'surprise',
    3: 'anger', 4: 'sadness', 5: 'disgust', 6: 'fear',
}
IEMOCAP_ID2EMOTION = {
    0: 'neutral', 1: 'happy', 2: 'sadness',
    3: 'anger', 4: 'frustrated',
}
EMOTION_EMOJI = {
    'neutral': '😐', 'joy': '😄', 'happy': '😊', 'surprise': '😲',
    'anger': '😠', 'sadness': '😢', 'disgust': '🤢',
    'fear': '😨', 'frustrated': '😤',
}
EMOTION_COLOR = {
    'neutral': '#94A3B8', 'joy': '#F59E0B', 'happy': '#FBBF24',
    'surprise': '#8B5CF6', 'anger': '#EF4444', 'sadness': '#3B82F6',
    'disgust': '#10B981', 'fear': '#EC4899', 'frustrated': '#F97316',
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()
        self.mel_transform   = T.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=80
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d((4, 16))
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * 4 * 16, hidden_size),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        mel = self.amplitude_to_db(self.mel_transform(waveform))
        mel = mel.unsqueeze(1)
        f   = self.cnn(mel).view(mel.size(0), -1)
        return self.projection(f)


class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers=7, embedding_dim=64, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(num_speakers, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, speaker_ids):
        return self.projection(self.embedding(speaker_ids))


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.3):
        super().__init__()
        self.text_to_audio = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_text = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1   = nn.LayerNorm(hidden_size)
        self.norm2   = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        t = text_feat.unsqueeze(1)
        a = audio_feat.unsqueeze(1)
        ta, _ = self.text_to_audio(t, a, a)
        at, _ = self.audio_to_text(a, t, t)
        return (self.norm1(t + self.dropout(ta)).squeeze(1),
                self.norm2(a + self.dropout(at)).squeeze(1))


class GatedFusion(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()
        self.gate    = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid()
        )
        self.fusion  = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        c = torch.cat([text_feat, audio_feat], dim=-1)
        return self.dropout(
            self.gate(c) * text_feat +
            (1 - self.gate(c)) * audio_feat +
            self.fusion(c)
        )


class EmotionModel(nn.Module):
    """Unified model for both MELD (7 emo, 7 spk) and IEMOCAP (5 emo, 10 spk)."""
    def __init__(self, num_emotions=7, num_speakers=7,
                 hidden_size=768, speaker_emb_dim=64, dropout=0.3):
        super().__init__()
        self.text_encoder      = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        self.audio_encoder     = AudioEncoder(hidden_size, dropout)
        self.speaker_embedding = SpeakerEmbedding(num_speakers, speaker_emb_dim, dropout)
        self.cross_attention   = CrossModalAttention(hidden_size, 8, dropout)
        self.gated_fusion      = GatedFusion(hidden_size, dropout)
        self.dropout           = nn.Dropout(dropout)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size + speaker_emb_dim, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask, waveform, speaker_ids):
        text  = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text  = self.dropout(text)
        audio = self.audio_encoder(waveform)
        ta, at = self.cross_attention(text, audio)
        fused  = self.dropout(self.gated_fusion(ta, at))
        spk    = self.speaker_embedding(speaker_ids)
        return self.emotion_head(torch.cat([fused, spk], dim=-1))


log.info(f"Loading models on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH, local_files_only=True)

log.info("  Loading MELD model...")
meld_model = EmotionModel(num_emotions=7, num_speakers=7).to(DEVICE).float()
meld_model.load_state_dict(
    torch.load(MELD_CKPT, map_location=DEVICE, weights_only=True),
    strict=False
)
meld_model.eval()
log.info("  ✅ MELD ready")

log.info("  Loading IEMOCAP model...")
iemocap_model = EmotionModel(num_emotions=5, num_speakers=10).to(DEVICE).float()
iemocap_model.load_state_dict(
    torch.load(IEMOCAP_CKPT, map_location=DEVICE, weights_only=True),
    strict=False
)
iemocap_model.eval()
log.info("  ✅ IEMOCAP ready")

MODELS = {
    'meld':    {'model': meld_model,    'id2emotion': MELD_ID2EMOTION,    'name': 'MELD'},
    'iemocap': {'model': iemocap_model, 'id2emotion': IEMOCAP_ID2EMOTION, 'name': 'IEMOCAP'},
}
log.info("✅ All models loaded!")


def text_to_tensor(text):
    enc = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                    truncation=True, return_tensors='pt')
    return enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)


def audio_to_tensor(audio_bytes):
    try:
        buf = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buf)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform  = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        target_len = 48000
        if waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
        return waveform.to(DEVICE)
    except Exception as e:
        log.warning(f"Audio failed: {e}")
        return torch.zeros(1, 48000).to(DEVICE)


def predict(model_key, text, audio_bytes=None):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    cfg     = MODELS[model_key]
    model   = cfg['model']
    id2emo  = cfg['id2emotion']

    input_ids, attention_mask = text_to_tensor(text)
    waveform    = audio_to_tensor(audio_bytes) if audio_bytes else torch.zeros(1, 48000).to(DEVICE)
    speaker_ids = torch.zeros(1, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, waveform, speaker_ids)
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_id   = int(np.argmax(probs))
    pred_name = id2emo[pred_id]

    return {
        'model':             cfg['name'],
        'predicted_emotion': pred_name,
        'emoji':             EMOTION_EMOJI.get(pred_name, '🎭'),
        'color':             EMOTION_COLOR.get(pred_name, '#2EC4B6'),
        'confidence':        float(probs[pred_id]),
        'probabilities':     {id2emo[i]: float(p) for i, p in enumerate(probs)}
    }


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multimodal Emotion Recognition</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0F1F3D; color: #F0F4F8; min-height: 100vh; }
    .header {
      background: #0F1F3D; border-bottom: 2px solid #2EC4B6;
      padding: 20px 40px; display: flex; align-items: center; gap: 16px;
    }
    .header h1 { font-size: 22px; color: #fff; }
    .header p  { font-size: 13px; color: #94A3B8; }
    .badge {
      background: #2EC4B6; color: #0F1F3D; font-size: 11px;
      font-weight: bold; padding: 3px 8px; border-radius: 4px;
    }
    .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
    .card { background: #1A3A6B; border-radius: 12px; padding: 28px; margin-bottom: 24px; }
    .card h2 { font-size: 16px; color: #2EC4B6; margin-bottom: 16px; }
    label { font-size: 13px; color: #94A3B8; display: block; margin-bottom: 6px; }
    .model-tabs {
      display: flex; gap: 8px; margin-bottom: 20px;
      background: #0F1F3D; padding: 6px; border-radius: 10px;
    }
    .model-tab {
      flex: 1; padding: 12px; text-align: center; cursor: pointer;
      border-radius: 8px; transition: all 0.2s;
    }
    .model-tab:hover { background: rgba(46,196,182,0.1); }
    .model-tab.active { background: #2EC4B6; color: #0F1F3D; font-weight: bold; }
    .model-tab .model-name { font-size: 15px; }
    .model-tab .model-desc { font-size: 11px; opacity: 0.8; margin-top: 2px; }
    textarea, input[type=file] {
      width: 100%; background: #0F1F3D; border: 1px solid #334155;
      border-radius: 8px; color: #F0F4F8; font-size: 14px; padding: 12px; resize: vertical;
    }
    textarea { min-height: 100px; }
    input[type=file] { padding: 8px; cursor: pointer; }
    .btn {
      background: #2EC4B6; color: #0F1F3D; border: none; border-radius: 8px;
      padding: 12px 32px; font-size: 15px; font-weight: bold; cursor: pointer;
      width: 100%; margin-top: 16px;
    }
    .btn:hover { opacity: 0.85; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .result-card {
      background: #0F1F3D; border-radius: 12px; padding: 28px;
      margin-top: 24px; border: 2px solid #2EC4B6; display: none;
    }
    .result-header { display: flex; justify-content: space-between; margin-bottom: 18px; }
    .model-tag {
      background: #2EC4B6; color: #0F1F3D; font-size: 11px;
      font-weight: bold; padding: 4px 10px; border-radius: 12px;
    }
    .result-top { display: flex; align-items: center; gap: 20px; margin-bottom: 24px; }
    .emotion-emoji { font-size: 64px; }
    .emotion-name  { font-size: 36px; font-weight: bold; text-transform: capitalize; }
    .confidence    { font-size: 14px; color: #94A3B8; margin-top: 4px; }
    .bar-label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
    .bar-bg { background: #1A3A6B; border-radius: 4px; height: 10px; margin-bottom: 10px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
    .loading { text-align: center; padding: 20px; color: #2EC4B6; display: none; }
    .error-msg {
      background: #7F1D1D; border-radius: 8px; padding: 12px;
      color: #FCA5A5; margin-top: 12px; display: none;
    }
    .api-section { background: #0A1628; border-radius: 8px; padding: 16px; margin-top: 12px; }
    .api-section h3 { font-size: 13px; color: #2EC4B6; margin-bottom: 8px; }
    pre { font-size: 12px; color: #94A3B8; overflow-x: auto; white-space: pre-wrap; }
    .note {
      background: rgba(255,159,28,0.1); border-left: 3px solid #FF9F1C;
      border-radius: 6px; padding: 12px 16px; margin-top: 12px;
      font-size: 12px; color: #FFB860;
    }
    .footer { text-align: center; color: #475569; font-size: 12px; padding: 40px 20px; }
  </style>
</head>
<body>

<div class="header">
  <div>
    <h1>🎭 Multimodal Emotion Recognition</h1>
    <p>HIT Shenzhen · Kamal Elgallouch · Supervised by Prof. Meishan Zhang</p>
  </div>
  <span class="badge">MELD + IEMOCAP</span>
</div>

<div class="container">

  <div class="card">
    <h2>📝 Input</h2>
    <label>Select Model</label>
    <div class="model-tabs">
      <div class="model-tab active" data-model="meld" onclick="selectModel('meld')">
        <div class="model-name">MELD</div>
        <div class="model-desc">TV Dialogue · 7 emotions</div>
      </div>
      <div class="model-tab" data-model="iemocap" onclick="selectModel('iemocap')">
        <div class="model-name">IEMOCAP</div>
        <div class="model-desc">Dyadic Sessions · 5 emotions</div>
      </div>
    </div>
    <div style="margin-bottom:16px">
      <label>Utterance Text (required)</label>
      <textarea id="textInput" placeholder="e.g. I can't believe you did that to me!"></textarea>
    </div>
    <div>
      <label>Audio File (optional — WAV/MP3, max 10s)</label>
      <input type="file" id="audioInput" accept=".wav,.mp3,.ogg,.flac">
    </div>
    <button class="btn" id="predictBtn" onclick="runPrediction()">🔍 Predict Emotion</button>
    <div class="error-msg" id="errorMsg"></div>
  </div>

  <div class="loading" id="loading">⏳ Running inference...</div>

  <div class="result-card" id="resultCard">
    <div class="result-header">
      <span class="model-tag" id="resModelTag">MELD</span>
    </div>
    <div class="result-top">
      <div class="emotion-emoji" id="resEmoji"></div>
      <div>
        <div class="emotion-name" id="resEmotion"></div>
        <div class="confidence" id="resConf"></div>
      </div>
    </div>
    <div id="barChart"></div>
  </div>

  <div class="card">
    <h2>🔌 REST API</h2>
    <div class="api-section">
      <h3>POST /predict</h3>
<pre>curl -X POST http://&lt;host&gt;:{{ port }}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"model": "meld", "text": "I am so happy today!", "audio_base64": null}'</pre>
    </div>
    <div class="api-section" style="margin-top:12px">
      <h3>Response</h3>
<pre>{
  "model": "MELD",
  "predicted_emotion": "joy",
  "emoji": "😄",
  "confidence": 0.8432,
  "probabilities": { "joy": 0.84, ... }
}</pre>
    </div>
    <div class="api-section" style="margin-top:12px">
      <h3>GET /health</h3>
<pre>curl http://&lt;host&gt;:{{ port }}/health
→ {"status": "ok", "models": ["MELD","IEMOCAP"], "device": "cuda"}</pre>
    </div>
    <div class="note">
      <b>Note on MOCAP:</b> The IEMOCAP + MOCAP model (Text + Audio + Motion Capture)
      is research-only and not exposed via this prototype, as it requires specialized
      motion capture sensor data that regular users cannot provide.
    </div>
  </div>

  <div class="footer">
    ModernBERT + CNN Audio + Cross-Modal Attention · 162M params · 2 datasets
  </div>
</div>

<script>
const EMOTION_COLORS = {{ colors | tojson }};
let currentModel = 'meld';

function selectModel(modelKey) {
  currentModel = modelKey;
  document.querySelectorAll('.model-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.model === modelKey);
  });
}

async function runPrediction() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) { showError('Please enter some text.'); return; }

  document.getElementById('predictBtn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('errorMsg').style.display = 'none';

  try {
    let audioB64 = null;
    const audioFile = document.getElementById('audioInput').files[0];
    if (audioFile) audioB64 = await fileToBase64(audioFile);

    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: currentModel, text: text, audio_base64: audioB64 })
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.error || ('Server error: ' + resp.status));
    }
    displayResult(await resp.json());
  } catch (err) {
    showError('Error: ' + err.message);
  } finally {
    document.getElementById('predictBtn').disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}

function displayResult(data) {
  document.getElementById('resModelTag').textContent = data.model;
  document.getElementById('resEmoji').textContent   = data.emoji;
  document.getElementById('resEmotion').textContent = data.predicted_emotion;
  document.getElementById('resEmotion').style.color = data.color;
  document.getElementById('resConf').textContent    = `Confidence: ${(data.confidence*100).toFixed(1)}%`;

  const chart = document.getElementById('barChart');
  chart.innerHTML = '';
  Object.entries(data.probabilities)
    .sort((a,b) => b[1]-a[1])
    .forEach(([emotion, prob]) => {
      const pct = (prob*100).toFixed(1);
      const color = EMOTION_COLORS[emotion] || '#2EC4B6';
      chart.innerHTML += `
        <div class="bar-label">
          <span style="text-transform:capitalize">${emotion}</span><span>${pct}%</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:${pct}%; background:${color}"></div></div>`;
    });
  document.getElementById('resultCard').style.display = 'block';
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg;
  el.style.display = 'block';
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
</script>
</body>
</html>
"""


app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(HTML, port=PORT, colors=EMOTION_COLOR)


@app.route('/health')
def health():
    return jsonify({
        'status':           'ok',
        'models':           [cfg['name'] for cfg in MODELS.values()],
        'device':           str(DEVICE),
        'meld_emotions':    list(MELD_ID2EMOTION.values()),
        'iemocap_emotions': list(IEMOCAP_ID2EMOTION.values()),
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    model_key = data.get('model', 'meld').lower()
    text      = data.get('text', '').strip()
    audio_b64 = data.get('audio_base64', None)

    if model_key not in MODELS:
        return jsonify({'error': f'Unknown model: {model_key}. Use "meld" or "iemocap"'}), 400
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    audio_bytes = None
    if audio_b64:
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            return jsonify({'error': 'Invalid audio_base64'}), 400

    try:
        return jsonify(predict(model_key, text, audio_bytes))
    except Exception as e:
        log.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    log.info("=" * 55)
    log.info("  Multimodal Emotion Recognition — Prototype")
    log.info(f"  Device  : {DEVICE}")
    log.info(f"  Models  : MELD (7) + IEMOCAP (5)")
    log.info(f"  URL     : http://localhost:{PORT}")
    log.info("=" * 55)
    app.run(host='0.0.0.0', port=PORT, debug=False)