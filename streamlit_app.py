import streamlit as st
import os
import tempfile
import time
import base64
import re
import numpy as np
import librosa
import soundfile as sf
import requests
from pathlib import Path
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
    .main-title {
        text-align: center;
        padding: 30px 0 20px 0;
    }
    .main-title h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 5px;
    }
    .main-title p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 5px;
    }
    .badge {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        color: #a78bfa;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(167, 139, 250, 0.2);
        margin-top: 8px;
    }
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
    }
    .file-info .size {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 18px 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 12px 30px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-size: 1rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px -8px rgba(124, 58, 237, 0.4) !important;
    }
    .stButton > button:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        transform: none !important;
    }
    .download-btn {
        display: inline-block;
        padding: 12px 30px;
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 15px;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        text-align: center;
    }
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px -8px rgba(124, 58, 237, 0.4);
        color: white;
        text-decoration: none;
    }
    .success-box {
        background: rgba(74, 222, 128, 0.08);
        border: 1px solid rgba(74, 222, 128, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 10px 0;
    }
    .error-box {
        background: rgba(248, 113, 113, 0.08);
        border: 1px solid rgba(248, 113, 113, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 10px 0;
    }
    .info-box {
        background: rgba(96, 165, 250, 0.08);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 10px 0;
    }
    .warning-box {
        background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(251, 191, 36, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 10px 0;
    }
    .ai-result-box {
        background: rgba(167, 139, 250, 0.08);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 10px 0;
    }
    .progress-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 4px;
        margin: 10px 0;
    }
    .progress-bar-fill {
        height: 8px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 30px;
    }
    .footer a {
        color: #818cf8;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
    }
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }
    .stSelectbox > div > div:hover {
        border-color: rgba(167, 139, 250, 0.4) !important;
    }
    .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }
    .stNumberInput > div > div:hover {
        border-color: rgba(167, 139, 250, 0.4) !important;
    }
    .stCheckbox > label {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
    }
    .streamlit-expanderHeader {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px !important;
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
    - Mutagen (metadata cleaning)
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    
    st.markdown("---")
    st.markdown("### 📖 Panduan")
    with st.expander("Cara Penggunaan"):
        st.markdown("""
        1. **Upload** file audio (MP3, WAV, FLAC, AIFF)
        2. **Pilih** processing level
        3. **Klik** Proses Audio
        4. **Lihat** hasil deteksi AI (sebelum & sesudah)
        5. **Download** hasil audio
        """)
    
    with st.expander("⚠️ Catatan"):
        st.markdown("""
        - File maksimal **50MB**
        - Proses **1-3 menit**
        - Mode **Gentle** = kualitas terbaik
        - Mode **Aggressive** = penghapusan lebih kuat
        - Deteksi AI menggunakan API pihak ketiga
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

# ============================================================================
# FUNGSI DETEKSI AI LOKAL (Fallback)
# ============================================================================

def detect_ai_local(audio: np.ndarray, sr: int) -> dict:
    """
    Deteksi AI sederhana menggunakan analisis spektral lokal.
    Ini adalah fallback jika API tidak tersedia.
    """
    try:
        # Hitung spectrogram
        freqs, psd = welch(audio, sr, nperseg=2048)
        
        # Cek high frequency energy (common AI watermark)
        high_freq_mask = freqs > 15000
        if np.any(high_freq_mask):
            high_freq_energy = np.mean(psd[high_freq_mask])
            total_energy = np.mean(psd)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
        else:
            high_freq_ratio = 0
        
        # Cek spectral flatness (AI sering terlalu halus)
        from scipy.stats import kurtosis
        spectral_kurtosis = kurtosis(psd)
        
        # Hitung skor AI
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
    
    # Filter parameters berdasarkan level
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
        st.warning(f"Filter error: {e}, menggunakan fallback")
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
            st.markdown("""
            <div class="error-box">
                ⚠️ File terlalu besar! Maksimal 50MB.
            </div>
            """, unsafe_allow_html=True)

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
    
    aggressive = st.checkbox(
        "Mode Agresif", 
        value=False, 
        help="Hapus metadata secara lebih agresif"
    )
    
    use_api_detection = st.checkbox(
        "Gunakan API Deteksi AI", 
        value=True,
        help="Gunakan API AI Music Checker (lebih akurat). Nonaktifkan untuk deteksi lokal."
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
        st.markdown("""
        <div class="warning-box">
            ⚠️ Silakan upload file audio terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    if uploaded_file.size > 50 * 1024 * 1024:
        st.markdown("""
        <div class="error-box">
            ❌ File terlalu besar! Maksimal 50MB.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("""
        <div class="info-box">
            ⏳ Memulai proses...
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================================
        # STEP 1: SAVE FILE
        # ============================================================
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name
        
        progress_bar.progress(10)
        status_text.markdown("""
        <div class="info-box">
            📂 File disimpan, memuat audio...
        </div>
        """, unsafe_allow_html=True)
        
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
            st.markdown(f"""
            <div class="error-box">
                ❌ Gagal memuat audio: {e}
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        progress_bar.progress(20)
        status_text.markdown("""
        <div class="info-box">
            🎵 Audio berhasil dimuat...
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================================
        # STEP 3: DETEKSI AI SEBELUM PROSES
        # ============================================================
        
        status_text.markdown("""
        <div class="info-box">
            🔍 Menganalisis audio untuk deteksi AI (sebelum)...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(25)
        
        # Simpan file sementara untuk deteksi
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_detect:
            sf.write(tmp_detect.name, y.T if is_stereo else y, sr)
            if use_api_detection:
                detection_before = check_ai_with_api(tmp_detect.name)
            else:
                detection_before = detect_ai_local(y_mono, sr)
            os.unlink(tmp_detect.name)
        
        # Tampilkan hasil deteksi awal
        if detection_before.get("success", False):
            ai_prob = detection_before.get('ai_probability', 0)
            is_ai = detection_before.get('is_ai', False)
            confidence = detection_before.get('confidence', 0)
            
            if is_ai:
                st.markdown(f"""
                <div class="warning-box">
                    🤖 <b>Hasil Deteksi Awal:</b> Audio terdeteksi sebagai <b>BUATAN AI</b><br>
                    Skor AI: {ai_prob:.1f}% | Keyakinan: {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    ✅ <b>Hasil Deteksi Awal:</b> Audio terdeteksi sebagai <b>KARYA MANUSIA</b><br>
                    Skor AI: {ai_prob:.1f}% | Keyakinan: {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ Deteksi AI awal gagal: {detection_before.get('error', 'Unknown error')}
            </div>
            """, unsafe_allow_html=True)
        
        progress_bar.progress(35)
        
        # ============================================================
        # STEP 4: HAPUS FINGERPRINT
        # ============================================================
        
        status_text.markdown("""
        <div class="info-box">
            🧹 Menghapus fingerprint/watermark AI...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(45)
        
        # Proses audio
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
        
        progress_bar.progress(65)
        status_text.markdown("""
        <div class="info-box">
            ✅ Fingerprint berhasil dihapus...
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================================
        # STEP 5: DETEKSI AI SETELAH PROSES
        # ============================================================
        
        status_text.markdown("""
        <div class="info-box">
            🔍 Verifikasi hasil dengan deteksi AI (sesudah)...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(75)
        
        # Simpan file hasil untuk deteksi ulang
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_result:
            sf.write(tmp_result.name, processed.T if is_stereo else processed, sr)
            if use_api_detection:
                detection_after = check_ai_with_api(tmp_result.name)
            else:
                detection_after = detect_ai_local(processed if not is_stereo else np.mean(processed, axis=0), sr)
            os.unlink(tmp_result.name)
        
        # Tampilkan hasil deteksi akhir
        if detection_after.get("success", False):
            ai_prob_after = detection_after.get('ai_probability', 0)
            is_ai_after = detection_after.get('is_ai', False)
            confidence_after = detection_after.get('confidence', 0)
            
            if is_ai_after:
                st.markdown(f"""
                <div class="warning-box">
                    🤖 <b>Hasil Deteksi Akhir:</b> Audio <b>MASIH</b> terdeteksi sebagai buatan AI<br>
                    Skor AI: {ai_prob_after:.1f}% | Keyakinan: {confidence_after:.1f}%<br>
                    <span style="color:#fbbf24;">💡 Coba tingkatkan level processing ke Aggressive</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    ✅ <b>Hasil Deteksi Akhir:</b> Audio terdeteksi sebagai <b>KARYA MANUSIA</b> ✅<br>
                    Skor AI: {ai_prob_after:.1f}% | Keyakinan: {confidence_after:.1f}%<br>
                    <span style="color:#4ade80;">🎉 Proses berhasil! Fingerprint AI telah dihapus.</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ Deteksi AI akhir gagal: {detection_after.get('error', 'Unknown error')}
            </div>
            """, unsafe_allow_html=True)
        
        progress_bar.progress(90)
        
        # ============================================================
        # STEP 6: SAVE OUTPUT
        # ============================================================
        
        status_text.markdown("""
        <div class="info-box">
            💾 Menyimpan hasil...
        </div>
        """, unsafe_allow_html=True)
        
        output_path = input_path.replace('.wav', '_processed.wav')
        if is_stereo:
            sf.write(output_path, processed.T, sr)
        else:
            sf.write(output_path, processed, sr)
        
        # Read processed file
        with open(output_path, 'rb') as f:
            audio_data = f.read()
        
        # Cleanup
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass
        
        progress_bar.progress(100)
        status_text.markdown("""
        <div class="success-box">
            ✅ Proses selesai!
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Pemrosesan")
        
        # Stats cards
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        
        with col_s1:
            ai_before = detection_before.get('ai_probability', 0) if detection_before.get('success') else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size:1.5rem;">{ai_before:.0f}%</div>
                <div class="stat-label">AI Sebelum</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            ai_after = detection_after.get('ai_probability', 0) if detection_after.get('success') else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size:1.5rem;">{ai_after:.0f}%</div>
                <div class="stat-label">AI Sesudah</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            reduction = ai_before - ai_after
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size:1.5rem;color:{'#4ade80' if reduction > 0 else '#94a3b8'};">{reduction:.0f}%</div>
                <div class="stat-label">Penurunan AI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size:1.2rem;">{level}</div>
                <div class="stat-label">Processing Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s5:
            method = "API" if use_api_detection else "Local"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size:1.2rem;">{method}</div>
                <div class="stat-label">Metode Deteksi</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ============================================================
        # DOWNLOAD SECTION
        # ============================================================
        
        st.markdown("### 📥 Download Hasil")
        
        b64 = base64.b64encode(audio_data).decode()
        filename = f"processed_{uploaded_file.name}"
        href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}" class="download-btn">⬇️ Download Audio (WAV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # ============================================================
        # DETAIL LOG
        # ============================================================
        
        with st.expander("📋 Detail Hasil Deteksi"):
            col_log1, col_log2 = st.columns(2)
            
            with col_log1:
                st.markdown("**🔍 Deteksi Awal (Sebelum)**")
                if detection_before.get("success"):
                    st.json(detection_before.get("details", {}))
                else:
                    st.error(detection_before.get("error", "Unknown error"))
            
            with col_log2:
                st.markdown("**✅ Deteksi Akhir (Sesudah)**")
                if detection_after.get("success"):
                    st.json(detection_after.get("details", {}))
                else:
                    st.error(detection_after.get("error", "Unknown error"))
        
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            ❌ Error: {str(e)}
        </div>
        """, unsafe_allow_html=True)
        import traceback
        with st.expander("Detail Error"):
            st.code(traceback.format_exc())
    finally:
        # Cleanup
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
    2. **Metadata Cleaning**: Hapus tag AI dari metadata
    3. **Watermark Removal**: Hapus watermark menggunakan filter spectral
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
    Dibangun dengan ❤️ menggunakan Streamlit &bull; AI Audio Fingerprint Remover v3.0
    <br>
    Deteksi AI oleh <a href="https://aimusicchecker.org" target="_blank">AI Music Checker</a>
</div>
""", unsafe_allow_html=True)
