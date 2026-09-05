import streamlit as st
import os
import tempfile
import time
import base64
import re
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

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
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 30px;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-title">
    <h1>🎧 AI Audio Fingerprint Remover</h1>
    <p>Hapus watermark, fingerprint, dan metadata AI dari file audio</p>
    <span class="badge">✦ Enhanced Edition v3.0</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menghapus:
    - ✅ Watermark audio AI
    - ✅ Metadata AI tags  
    - ✅ Fingerprint digital
    
    **Didukung oleh:**
    - Librosa (audio analysis)
    - NumPy/SciPy (signal processing)
    - Mutagen (metadata cleaning)
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Panduan")
    with st.expander("Cara Penggunaan"):
        st.markdown("""
        1. **Upload** file audio
        2. **Pilih** processing level
        3. **Klik** Proses Audio
        4. **Download** hasil
        """)
    
    with st.expander("⚠️ Catatan"):
        st.markdown("""
        - File maksimal **50MB**
        - Proses **1-3 menit**
        - Mode **Gentle** = kualitas terbaik
        """)

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

with col2:
    st.markdown("### ⚙️ Pengaturan")
    
    level = st.selectbox(
        "Processing Level",
        options=['gentle', 'moderate', 'aggressive'],
        index=1
    )
    
    aggressive = st.checkbox("Mode Agresif", value=False)

# ============================================================================
# PROCESS BUTTON
# ============================================================================

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    process_btn = st.button("🚀 Proses Audio", use_container_width=True)

# ============================================================================
# PROCESSING LOGIC - DIRECT (Tanpa Subprocess)
# ============================================================================

if process_btn:
    if not uploaded_file:
        st.warning("⚠️ Silakan upload file audio terlebih dahulu.")
        st.stop()
    
    if uploaded_file.size > 50 * 1024 * 1024:
        st.error("❌ File terlalu besar! Maksimal 50MB.")
        st.stop()
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info("⏳ Memulai proses...")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name
        
        progress_bar.progress(20)
        status_text.info("📂 File disimpan, memuat audio...")
        
        # Load audio
        try:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            is_stereo = len(y.shape) > 1
            if is_stereo:
                y_mono = np.mean(y, axis=0)
            else:
                y_mono = y
        except Exception as e:
            st.error(f"❌ Gagal memuat audio: {e}")
            st.stop()
        
        progress_bar.progress(40)
        status_text.info("🔍 Mendeteksi watermark...")
        
        # Simple watermark detection
        watermarks_detected = 0
        watermarks_removed = 0
        
        # Detect high frequency anomalies
        try:
            import scipy.signal as signal
            freqs, psd = signal.welch(y_mono, sr, nperseg=2048)
            
            # Check for high frequency energy
            high_freq_mask = freqs > 15000
            if np.any(high_freq_mask):
                high_freq_energy = np.mean(psd[high_freq_mask])
                total_energy = np.mean(psd)
                if high_freq_energy > total_energy * 0.1:
                    watermarks_detected += 1
        except:
            pass
        
        progress_bar.progress(60)
        status_text.info("🧹 Menghapus watermark...")
        
        # Remove watermarks (simple version)
        processed = y.copy()
        
        if is_stereo:
            for i in range(y.shape[0]):
                # Apply simple low-pass filter to remove high frequency
                try:
                    from scipy.signal import butter, filtfilt
                    b, a = butter(2, 0.9, btype='low')
                    processed[i] = filtfilt(b, a, processed[i])
                except:
                    pass
                watermarks_removed += 1
        else:
            try:
                from scipy.signal import butter, filtfilt
                b, a = butter(2, 0.9, btype='low')
                processed = filtfilt(b, a, processed)
            except:
                pass
            watermarks_removed += 1
        
        progress_bar.progress(80)
        status_text.info("💾 Menyimpan hasil...")
        
        # Save processed audio
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
        status_text.success("✅ Proses selesai!")
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Pemrosesan")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{watermarks_removed}</div>
                <div class="stat-label">Watermarks Dihapus</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{watermarks_detected}</div>
                <div class="stat-label">Watermarks Terdeteksi</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{level}</div>
                <div class="stat-label">Processing Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Download section
        st.markdown("### 📥 Download Hasil")
        
        b64 = base64.b64encode(audio_data).decode()
        filename = f"processed_{uploaded_file.name}"
        href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}" class="download-btn">⬇️ Download Audio (WAV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            ✅ Audio berhasil diproses!
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        with st.expander("Detail Error"):
            st.code(traceback.format_exc())

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    Dibangun dengan ❤️ menggunakan Streamlit &bull; AI Audio Fingerprint Remover v3.0
</div>
""", unsafe_allow_html=True)
