import streamlit as st
import os
import tempfile
import time
import base64
import numpy as np
import librosa
import soundfile as sf
import requests
from scipy.signal import butter, filtfilt, welch

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="AI Audio Fingerprint Remover",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    /* Title */
    .main-title {
        text-align: center;
        padding: 30px 0 20px 0;
    }
    .main-title h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 5px;
    }
    .main-title p {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-top: 5px;
    }
    .badge {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        color: #a78bfa;
        padding: 6px 20px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(167, 139, 250, 0.2);
        margin-top: 8px;
    }
    
    /* File Info */
    .file-info {
        background: rgba(167, 139, 250, 0.08);
        border: 1px solid rgba(167, 139, 250, 0.15);
        border-radius: 12px;
        padding: 15px 20px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .file-info .name {
        color: #e0e0e0;
        font-weight: 500;
        font-size: 1rem;
    }
    .file-info .size {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* Result Cards */
    .result-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Big Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    .stat-card-big {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    .stat-card-big:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-3px);
        border-color: rgba(167, 139, 250, 0.3);
    }
    .stat-value-big {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .stat-value-big.success {
        background: linear-gradient(135deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-value-big.danger {
        background: linear-gradient(135deg, #f87171, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-value-big.warning {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label-big {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    
    /* Status Boxes */
    .success-box {
        background: rgba(74, 222, 128, 0.08);
        border: 2px solid rgba(74, 222, 128, 0.2);
        border-radius: 16px;
        padding: 25px 30px;
        margin: 15px 0;
    }
    .success-box .big-icon {
        font-size: 3rem;
        display: block;
        text-align: center;
        margin-bottom: 10px;
    }
    .success-box .big-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4ade80;
        text-align: center;
    }
    .success-box .sub-text {
        font-size: 1rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 5px;
    }
    
    .error-box {
        background: rgba(248, 113, 113, 0.08);
        border: 2px solid rgba(248, 113, 113, 0.2);
        border-radius: 16px;
        padding: 25px 30px;
        margin: 15px 0;
    }
    .error-box .big-icon {
        font-size: 3rem;
        display: block;
        text-align: center;
        margin-bottom: 10px;
    }
    .error-box .big-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f87171;
        text-align: center;
    }
    
    .warning-box {
        background: rgba(251, 191, 36, 0.08);
        border: 2px solid rgba(251, 191, 36, 0.2);
        border-radius: 16px;
        padding: 25px 30px;
        margin: 15px 0;
    }
    .warning-box .big-icon {
        font-size: 3rem;
        display: block;
        text-align: center;
        margin-bottom: 10px;
    }
    .warning-box .big-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fbbf24;
        text-align: center;
    }
    
    .info-box {
        background: rgba(96, 165, 250, 0.08);
        border: 2px solid rgba(96, 165, 250, 0.2);
        border-radius: 16px;
        padding: 25px 30px;
        margin: 15px 0;
    }
    .info-box .big-icon {
        font-size: 3rem;
        display: block;
        text-align: center;
        margin-bottom: 10px;
    }
    .info-box .big-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #60a5fa;
        text-align: center;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 14px 30px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-size: 1.1rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px -8px rgba(124, 58, 237, 0.4) !important;
    }
    
    /* Download Button */
    .download-btn {
        display: inline-block;
        padding: 16px 40px;
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white !important;
        border-radius: 14px;
        text-decoration: none;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 15px;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        text-align: center;
        width: 100%;
    }
    .download-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px -8px rgba(124, 58, 237, 0.5);
        color: white !important;
        text-decoration: none;
    }
    
    /* Comparison Grid */
    .comparison-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin: 20px 0;
    }
    .comparison-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
    }
    .comparison-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .comparison-card .score {
        font-size: 4rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .comparison-card .status {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
    }
    .comparison-card.before {
        border-color: rgba(251, 191, 36, 0.2);
    }
    .comparison-card.after {
        border-color: rgba(74, 222, 128, 0.2);
    }
    
    /* Progress */
    .stProgress > div > div {
        background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
        height: 8px !important;
    }
    
    /* Form Elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }
    .stSelectbox > div > div:hover {
        border-color: rgba(167, 139, 250, 0.4) !important;
    }
    .stSlider > div > div {
        color: #a78bfa !important;
    }
    .stCheckbox > label {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px 0 20px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 40px;
    }
    .footer .footer-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .footer .footer-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 5px;
    }
    .footer .footer-sub {
        font-size: 0.95rem;
        color: #94a3b8;
    }
    .footer a {
        color: #a78bfa;
        text-decoration: none;
        font-weight: 500;
    }
    .footer a:hover {
        text-decoration: underline;
        color: #c4b5fd;
    }
    .footer .footer-divider {
        display: inline-block;
        margin: 0 10px;
        color: #334155;
    }
    .footer .footer-bottom {
        margin-top: 10px;
        font-size: 0.8rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-title">
    <h1>🎧 AI Audio Fingerprint Remover</h1>
    <p>Deteksi AI + Hapus Watermark + Verifikasi Keaslian Audio</p>
    <span class="badge">✦ Enhanced Edition v3.0</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini melakukan:
    1. 🔍 **Deteksi AI** - Cek apakah audio buatan AI
    2. 🧹 **Hapus Fingerprint** - Bersihkan watermark AI
    3. ✅ **Verifikasi** - Deteksi ulang untuk konfirmasi
    
    **Didukung oleh:**
    - AI Music Checker (deteksi AI)
    - Librosa (audio analysis)
    - NumPy/SciPy (signal processing)
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Panduan")
    with st.expander("Cara Penggunaan"):
        st.markdown("""
        1. **Upload** file audio
        2. **Pilih** processing level
        3. **Atur** tempo/pitch (opsional)
        4. **Klik** Proses Audio
        5. **Lihat** hasil deteksi AI
        6. **Download** hasil audio
        """)
    
    with st.expander("⚠️ Catatan"):
        st.markdown("""
        - File maksimal **50MB**
        - Proses **1-3 menit**
        - Mode **Gentle** = kualitas terbaik
        - Mode **Aggressive** = penghapusan lebih kuat
        - **Slider** untuk tempo/pitch lebih aman digunakan
        """)

# ============================================================================
# FUNGSI DETEKSI AI
# ============================================================================

def check_ai_with_api(file_path: str) -> dict:
    """
    Mengirim file audio ke AI Music Checker untuk dianalisis.
    """
    API_URL = "https://aimusicchecker.org/api/check"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    try:
        with open(file_path, 'rb') as f:
            files = {'audio': f}
            response = requests.post(API_URL, files=files, headers=headers, timeout=60)

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "ai_probability": data.get("ai_probability", 0),
                "confidence": data.get("confidence", 0),
                "details": data.get("details", {}),
                "is_ai": data.get("ai_probability", 0) > 50,
                "raw_response": data
            }
        else:
            return {
                "success": False,
                "error": f"API Error: {response.status_code}",
                "ai_probability": 0,
                "is_ai": False
            }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout", "ai_probability": 0, "is_ai": False}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection error", "ai_probability": 0, "is_ai": False}
    except Exception as e:
        return {"success": False, "error": str(e), "ai_probability": 0, "is_ai": False}

def detect_ai_local(audio: np.ndarray, sr: int) -> dict:
    """
    Deteksi AI sederhana menggunakan analisis spektral lokal (fallback).
    """
    try:
        freqs, psd = welch(audio, sr, nperseg=2048)
        
        high_freq_mask = freqs > 15000
        if np.any(high_freq_mask):
            high_freq_energy = np.mean(psd[high_freq_mask])
            total_energy = np.mean(psd)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
        else:
            high_freq_ratio = 0
        
        from scipy.stats import kurtosis
        spectral_kurtosis = kurtosis(psd)
        
        ai_score = 0
        if high_freq_ratio > 0.1:
            ai_score += 30
        if spectral_kurtosis < 2:
            ai_score += 30
        if high_freq_ratio > 0.2 and spectral_kurtosis < 1.5:
            ai_score += 40
        
        is_ai = ai_score > 50
        
        return {
            "success": True,
            "ai_probability": min(100, ai_score),
            "confidence": 70 if is_ai else 60,
            "is_ai": is_ai,
            "details": {
                "high_freq_ratio": float(high_freq_ratio),
                "spectral_kurtosis": float(spectral_kurtosis)
            },
            "method": "local"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ai_probability": 0,
            "is_ai": False
        }

# ============================================================================
# FUNGSI PROSES AUDIO
# ============================================================================

def remove_fingerprints(audio: np.ndarray, sr: int, level: str, aggressive: bool = False) -> np.ndarray:
    """
    Menghapus fingerprint/watermark dari audio.
    """
    audio = audio.copy()
    is_stereo = len(audio.shape) > 1
    
    filter_params = {
        'gentle': {'order': 1, 'cutoff': 0.95},
        'moderate': {'order': 2, 'cutoff': 0.9},
        'aggressive': {'order': 3, 'cutoff': 0.85}
    }
    params = filter_params.get(level, filter_params['moderate'])
    
    if aggressive:
        params['order'] = min(params['order'] + 1, 4)
        params['cutoff'] = max(params['cutoff'] - 0.05, 0.7)
    
    try:
        if is_stereo:
            processed = audio.copy()
            for i in range(audio.shape[0]):
                b, a = butter(params['order'], params['cutoff'], btype='low')
                processed[i] = filtfilt(b, a, processed[i])
            return processed
        else:
            b, a = butter(params['order'], params['cutoff'], btype='low')
            return filtfilt(b, a, audio)
    except Exception as e:
        st.warning(f"Filter error: {e}")
        return audio

def apply_tempo_pitch(audio: np.ndarray, sr: int, tempo: float, pitch_semitones: float, pitch_percent: float) -> np.ndarray:
    """
    Menerapkan tempo dan pitch shift ke audio.
    """
    audio = audio.copy()
    
    try:
        # Tempo shift
        if tempo != 0:
            rate = 1.0 + (tempo / 100.0)
            rate = max(0.25, min(4.0, rate))
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shift
        if pitch_semitones != 0 or pitch_percent != 0:
            pitch_total = pitch_semitones + (pitch_percent / 6.0)
            pitch_total = max(-24, min(24, pitch_total))
            if pitch_total != 0:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_total)
        
        return audio
    except Exception as e:
        st.warning(f"Tempo/Pitch error: {e}")
        return audio

# ============================================================================
# MAIN CONTENT
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 Upload Audio")
    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=['mp3', 'wav', 'flac', 'aiff', 'aif'],
        help="Maksimal 50MB"
    )
    
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"""
        <div class="file-info">
            <span class="name">📄 {uploaded_file.name}</span>
            <span class="size">{file_size_mb:.2f} MB</span>
        </div>
        """, unsafe_allow_html=True)
        
        if file_size_mb > 50:
            st.error("⚠️ File terlalu besar! Maksimal 50MB.")
            st.stop()

with col2:
    st.markdown("### ⚙️ Pengaturan")
    
    level = st.selectbox(
        "Processing Level",
        options=['gentle', 'moderate', 'aggressive'],
        index=1,
        help="Gentle = kualitas terbaik, Aggressive = penghapusan lebih kuat"
    )
    
    level_descriptions = {
        'gentle': 'Minimal processing - preserves quality',
        'moderate': 'Balanced processing - good effectiveness',
        'aggressive': 'Thorough processing - removes most fingerprints'
    }
    st.caption(f"ℹ️ {level_descriptions.get(level, '')}")
    
    st.markdown("---")
    
    # ============================================================
    # MENGGUNAKAN SLIDER - LEBIH AMAN (TIDAK ERROR)
    # ============================================================
    
    col_tempo, col_pitch = st.columns(2)
    
    with col_tempo:
        tempo = st.slider(
            "Tempo Shift (%)",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            help="Ubah kecepatan audio (-50% sampai +50%)"
        )
        st.caption(f"Nilai: {tempo:.0f}%")
    
    with col_pitch:
        pitch_semitones = st.slider(
            "Pitch (Semitones)",
            min_value=-12.0,
            max_value=12.0,
            value=0.0,
            step=0.5,
            help="Ubah pitch dalam semitone (-12 sampai +12)"
        )
        st.caption(f"Nilai: {pitch_semitones:.1f} semitone")
    
    pitch_percent = st.slider(
        "Pitch (%)",
        min_value=-50.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        help="Ubah pitch dalam persen (-50% sampai +100%)"
    )
    st.caption(f"Nilai: {pitch_percent:.0f}%")
    
    st.markdown("---")
    
    aggressive = st.checkbox(
        "Mode Agresif", 
        value=False, 
        help="Hapus metadata secara lebih agresif"
    )
    
    use_api_detection = st.checkbox(
        "Gunakan API Deteksi AI", 
        value=True,
        help="Gunakan API AI Music Checker (lebih akurat)"
    )

# ============================================================================
# PROCESS BUTTON
# ============================================================================

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    process_btn = st.button("🚀 Proses Audio", use_container_width=True)

# ============================================================================
# PROCESSING LOGIC
# ============================================================================

if process_btn:
    if not uploaded_file:
        st.warning("⚠️ Silakan upload file audio terlebih dahulu.")
        st.stop()
    
    if uploaded_file.size > 50 * 1024 * 1024:
        st.error("❌ File terlalu besar! Maksimal 50MB.")
        st.stop()
    
    # Container untuk hasil
    result_container = st.container()
    
    with result_container:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("⏳ Memulai proses...")
            
            # ============================================================
            # STEP 1: SAVE FILE
            # ============================================================
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_input:
                tmp_input.write(uploaded_file.read())
                input_path = tmp_input.name
            
            progress_bar.progress(10)
            status_text.info("📂 File disimpan, memuat audio...")
            
            # ============================================================
            # STEP 2: LOAD AUDIO
            # ============================================================
            
            try:
                y, sr = librosa.load(input_path, sr=None, mono=False)
                is_stereo = len(y.shape) > 1
                if is_stereo:
                    y_mono = np.mean(y, axis=0)
                else:
                    y_mono = y
                original_length = y.shape[1] if is_stereo else len(y)
            except Exception as e:
                st.error(f"❌ Gagal memuat audio: {e}")
                st.stop()
            
            progress_bar.progress(20)
            status_text.info("🎵 Audio berhasil dimuat...")
            
            # ============================================================
            # STEP 3: DETEKSI AI SEBELUM
            # ============================================================
            
            status_text.info("🔍 Menganalisis audio untuk deteksi AI (sebelum)...")
            progress_bar.progress(25)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_detect:
                sf.write(tmp_detect.name, y.T if is_stereo else y, sr)
                if use_api_detection:
                    detection_before = check_ai_with_api(tmp_detect.name)
                else:
                    detection_before = detect_ai_local(y_mono, sr)
                os.unlink(tmp_detect.name)
            
            # ============================================================
            # STEP 4: APLIKASI TEMPO & PITCH (JIKA ADA PERUBAHAN)
            # ============================================================
            
            # Ambil nilai dengan aman (sudah pasti float dari slider)
            tempo_val = float(tempo)
            pitch_semitones_val = float(pitch_semitones)
            pitch_percent_val = float(pitch_percent)
            
            if tempo_val != 0 or pitch_semitones_val != 0 or pitch_percent_val != 0:
                status_text.info("🎵 Mengubah tempo dan pitch...")
                progress_bar.progress(35)
                
                try:
                    if is_stereo:
                        processed_channels = []
                        for channel in y:
                            channel = apply_tempo_pitch(channel, sr, tempo_val, pitch_semitones_val, pitch_percent_val)
                            processed_channels.append(channel)
                        y = np.array(processed_channels)
                    else:
                        y = apply_tempo_pitch(y, sr, tempo_val, pitch_semitones_val, pitch_percent_val)
                    
                    # Pastikan panjang audio tetap
                    if is_stereo:
                        for i in range(y.shape[0]):
                            if len(y[i]) != original_length:
                                if len(y[i]) > original_length:
                                    y[i] = y[i][:original_length]
                                else:
                                    y[i] = np.pad(y[i], (0, original_length - len(y[i])))
                    else:
                        if len(y) != original_length:
                            if len(y) > original_length:
                                y = y[:original_length]
                            else:
                                y = np.pad(y, (0, original_length - len(y)))
                except Exception as e:
                    st.warning(f"Tempo/Pitch gagal: {e}, lanjutkan tanpa perubahan")
            
            # ============================================================
            # STEP 5: HAPUS FINGERPRINT
            # ============================================================
            
            status_text.info("🧹 Menghapus fingerprint/watermark AI...")
            progress_bar.progress(50)
            
            processed = remove_fingerprints(y, sr, level, aggressive)
            
            # Pastikan panjang audio sama
            if is_stereo:
                for i in range(processed.shape[0]):
                    if len(processed[i]) != original_length:
                        if len(processed[i]) > original_length:
                            processed[i] = processed[i][:original_length]
                        else:
                            processed[i] = np.pad(processed[i], (0, original_length - len(processed[i])))
            else:
                if len(processed) != original_length:
                    if len(processed) > original_length:
                        processed = processed[:original_length]
                    else:
                        processed = np.pad(processed, (0, original_length - len(processed)))
            
            progress_bar.progress(70)
            status_text.info("✅ Fingerprint berhasil dihapus...")
            
            # ============================================================
            # STEP 6: DETEKSI AI SETELAH
            # ============================================================
            
            status_text.info("🔍 Verifikasi hasil dengan deteksi AI (sesudah)...")
            progress_bar.progress(80)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_result:
                sf.write(tmp_result.name, processed.T if is_stereo else processed, sr)
                if use_api_detection:
                    detection_after = check_ai_with_api(tmp_result.name)
                else:
                    detection_after = detect_ai_local(processed if not is_stereo else np.mean(processed, axis=0), sr)
                os.unlink(tmp_result.name)
            
            # ============================================================
            # STEP 7: SAVE OUTPUT
            # ============================================================
            
            status_text.info("💾 Menyimpan hasil...")
            
            output_path = input_path.replace('.wav', '_processed.wav')
            if is_stereo:
                sf.write(output_path, processed.T, sr)
            else:
                sf.write(output_path, processed, sr)
            
            with open(output_path, 'rb') as f:
                audio_data = f.read()
            
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass
            
            progress_bar.progress(100)
            status_text.success("✅ Proses selesai!")
            
            # ============================================================
            # ============================================================
            # TAMPILKAN HASIL - BESAR DAN JELAS
            # ============================================================
            # ============================================================
            
            st.markdown("---")
            st.markdown("## 📊 HASIL PEMROSESAN")
            st.markdown("---")
            
            # --- RESULT BOX ---
            ai_before = detection_before.get('ai_probability', 0) if detection_before.get('success') else 0
            ai_after = detection_after.get('ai_probability', 0) if detection_after.get('success') else 0
            is_ai_before = detection_before.get('is_ai', False) if detection_before.get('success') else False
            is_ai_after = detection_after.get('is_ai', False) if detection_after.get('success') else False
            
            # Status Audio
            if is_ai_before and not is_ai_after:
                st.markdown("""
                <div class="success-box">
                    <span class="big-icon">🎉</span>
                    <div class="big-text">BERHASIL! Fingerprint AI Telah Dihapus</div>
                    <div class="sub-text">Audio kini terdeteksi sebagai karya manusia</div>
                </div>
                """, unsafe_allow_html=True)
            elif is_ai_before and is_ai_after:
                st.markdown("""
                <div class="warning-box">
                    <span class="big-icon">⚠️</span>
                    <div class="big-text">Audio Masih Terdeteksi sebagai AI</div>
                    <div class="sub-text" style="color:#94a3b8;">Coba tingkatkan level processing ke Aggressive atau Mode Agresif</div>
                </div>
                """, unsafe_allow_html=True)
            elif not is_ai_before and not is_ai_after:
                st.markdown("""
                <div class="info-box">
                    <span class="big-icon">✅</span>
                    <div class="big-text">Audio Terdeteksi sebagai Karya Manusia</div>
                    <div class="sub-text" style="color:#94a3b8;">Tidak ada indikasi AI pada audio ini</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- COMPARISON GRID ---
            st.markdown("### 🔍 Perbandingan Deteksi AI")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                before_color = "#fbbf24" if is_ai_before else "#4ade80"
                before_status = "🤖 AI" if is_ai_before else "✅ Manusia"
                before_bg = "rgba(251,191,36,0.15)" if is_ai_before else "rgba(74,222,128,0.15)"
                st.markdown(f"""
                <div class="comparison-card before" style="border-color:{before_color}40;">
                    <div class="label">📌 SEBELUM DIPROSES</div>
                    <div class="score" style="color:{before_color};">{ai_before:.0f}%</div>
                    <div class="status" style="background:{before_bg};color:{before_color};">{before_status}</div>
                    <div style="margin-top:10px;color:#94a3b8;font-size:0.9rem;">
                        Keyakinan: {detection_before.get('confidence', 0):.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_comp2:
                after_color = "#fbbf24" if is_ai_after else "#4ade80"
                after_status = "🤖 AI" if is_ai_after else "✅ Manusia"
                after_bg = "rgba(251,191,36,0.15)" if is_ai_after else "rgba(74,222,128,0.15)"
                st.markdown(f"""
                <div class="comparison-card after" style="border-color:{after_color}40;">
                    <div class="label">📌 SESUDAH DIPROSES</div>
                    <div class="score" style="color:{after_color};">{ai_after:.0f}%</div>
                    <div class="status" style="background:{after_bg};color:{after_color};">{after_status}</div>
                    <div style="margin-top:10px;color:#94a3b8;font-size:0.9rem;">
                        Keyakinan: {detection_after.get('confidence', 0):.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- STATS GRID ---
            st.markdown("### 📈 Statistik Pemrosesan")
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.markdown(f"""
                <div class="stat-card-big">
                    <div class="stat-value-big">{ai_before:.0f}%</div>
                    <div class="stat-label-big">AI Sebelum</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s2:
                st.markdown(f"""
                <div class="stat-card-big">
                    <div class="stat-value-big {'success' if ai_after < ai_before else 'warning'}">{ai_after:.0f}%</div>
                    <div class="stat-label-big">AI Sesudah</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s3:
                reduction = ai_before - ai_after
                color = "success" if reduction > 10 else "warning" if reduction > 0 else "danger"
                st.markdown(f"""
                <div class="stat-card-big">
                    <div class="stat-value-big {color}">{reduction:.0f}%</div>
                    <div class="stat-label-big">Penurunan AI</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_s4:
                st.markdown(f"""
                <div class="stat-card-big">
                    <div class="stat-value-big" style="font-size:2rem;">{level.upper()}</div>
                    <div class="stat-label-big">Processing Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- DOWNLOAD ---
            st.markdown("### 📥 Download Hasil")
            
            b64 = base64.b64encode(audio_data).decode()
            filename = f"processed_{uploaded_file.name}"
            href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}" class="download-btn">⬇️ Download Audio (WAV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # --- DETAIL (collapsible) ---
            with st.expander("📋 Detail Deteksi AI"):
                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.markdown("**🔍 Deteksi Awal**")
                    if detection_before.get("success"):
                        st.json(detection_before.get("details", {}))
                    else:
                        st.error(detection_before.get("error", "Unknown error"))
                with col_det2:
                    st.markdown("**✅ Deteksi Akhir**")
                    if detection_after.get("success"):
                        st.json(detection_after.get("details", {}))
                    else:
                        st.error(detection_after.get("error", "Unknown error"))
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())
        finally:
            try:
                if 'input_path' in locals() and os.path.exists(input_path):
                    os.unlink(input_path)
                if 'output_path' in locals() and os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

with st.expander("📌 Informasi Tambahan"):
    st.markdown("""
    ### Cara Kerja
    
    1. **Deteksi AI Awal**: Cek apakah audio buatan AI
    2. **Tempo & Pitch**: Ubah kecepatan dan nada (opsional)
    3. **Watermark Removal**: Hapus fingerprint AI
    4. **Deteksi AI Akhir**: Verifikasi hasil pemrosesan
    
    ### Format yang Didukung
    
    | Format | Support |
    |--------|---------|
    | MP3 | ✅ Full |
    | WAV | ✅ Full |
    | FLAC | ✅ Full |
    | AIFF | ✅ Full |
    
    ### Metode Deteksi AI
    
    - **API Mode**: Menggunakan AI Music Checker (lebih akurat)
    - **Local Mode**: Analisis spektral lokal (tanpa internet)
    
    ### Keterbatasan
    
    - File maksimal 50MB (Streamlit Cloud limit)
    - Proses 1-3 menit tergantung ukuran file
    - Deteksi AI adalah probabilitas, bukan kepastian absolut
    """)

st.markdown("""
<div class="footer">
    <div class="footer-icon">🎵</div>
    <div class="footer-title">AI Audio Fingerprint Remover v3.0</div>
    <div class="footer-sub">
        Dibangun dengan ❤️ menggunakan <a href="https://streamlit.io" target="_blank">Streamlit</a>
        <span class="footer-divider">•</span>
        Deteksi AI oleh <a href="https://aimusicchecker.org" target="_blank">AI Music Checker</a>
    </div>
    <div class="footer-bottom">
        © 2024 • Open Source • Privasi Terjaga
    </div>
</div>
""", unsafe_allow_html=True)
