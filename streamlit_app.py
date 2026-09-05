import streamlit as st
import os
import tempfile
import time
import subprocess
import sys
import base64
import re
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
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    /* Title styling */
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
    
    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(167, 139, 250, 0.3);
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.02);
    }
    .upload-area:hover {
        border-color: rgba(167, 139, 250, 0.6);
        background: rgba(167, 139, 250, 0.05);
    }
    .upload-area .icon {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .upload-area .text {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    /* File info */
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
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 15px 0;
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
    
    /* Buttons */
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
    
    /* Download button */
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
    
    /* Status boxes */
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
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }
    .stSelectbox > div > div:hover {
        border-color: rgba(167, 139, 250, 0.4) !important;
    }
    
    /* Number input */
    .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }
    .stNumberInput > div > div:hover {
        border-color: rgba(167, 139, 250, 0.4) !important;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px !important;
    }
    
    /* Footer */
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
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
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
    - ✅ Pattern AI yang terdeteksi
    
    **Didukung oleh:**
    - Librosa (audio analysis)
    - NumPy/SciPy (signal processing)
    - Mutagen (metadata cleaning)
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    status_placeholder = st.empty()
    
    st.markdown("---")
    st.markdown("### 📖 Panduan")
    with st.expander("Cara Penggunaan"):
        st.markdown("""
        1. **Upload** file audio (MP3, WAV, FLAC, AIFF)
        2. **Pilih** processing level yang diinginkan
        3. **Atur** tempo/pitch jika perlu
        4. **Klik** tombol Proses Audio
        5. **Tunggu** hingga proses selesai
        6. **Download** hasil audio
        """)
    
    with st.expander("⚠️ Catatan"):
        st.markdown("""
        - File maksimal **50MB**
        - Proses bisa memakan waktu **1-5 menit**
        - Mode **Gentle** = kualitas terbaik
        - Mode **Extreme** = penghapusan maksimum
        """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("### 📁 Upload Audio")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=['mp3', 'wav', 'flac', 'aiff', 'aif'],
        help="Maksimal 50MB. Format: MP3, WAV, FLAC, AIFF"
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
    else:
        st.markdown("""
        <div class="upload-area">
            <div class="icon">🎵</div>
            <div class="text">Drag & drop atau klik untuk upload</div>
            <div style="color:#64748b;font-size:0.8rem;margin-top:5px;">MP3, WAV, FLAC, AIFF</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### ⚙️ Pengaturan")
    
    level = st.selectbox(
        "Processing Level",
        options=['gentle', 'moderate', 'aggressive', 'extreme'],
        index=1,
        help="Gentle = kualitas terbaik, Extreme = penghapusan maksimum"
    )
    
    level_descriptions = {
        'gentle': 'Minimal processing - preserves quality',
        'moderate': 'Balanced processing - good effectiveness',
        'aggressive': 'Thorough processing - removes most fingerprints',
        'extreme': 'Maximum processing - may affect quality'
    }
    st.caption(f"ℹ️ {level_descriptions.get(level, '')}")
    
    st.markdown("---")
    
    col_tempo, col_pitch = st.columns(2)
    
    with col_tempo:
        tempo = st.number_input(
            "Tempo Shift (%)",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            help="Ubah kecepatan audio (-50% sampai +50%)"
        )
    
    with col_pitch:
        pitch_semitones = st.number_input(
            "Pitch (Semitones)",
            min_value=-12.0,
            max_value=12.0,
            value=0.0,
            step=0.5,
            help="Ubah pitch dalam semitone (-12 sampai +12)"
        )
    
    pitch_percent = st.number_input(
        "Pitch (%)",
        min_value=-50.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        help="Ubah pitch dalam persen (-50% sampai +100%)"
    )
    
    aggressive = st.checkbox(
        "Mode Agresif", 
        value=False, 
        help="Hapus semua metadata secara agresif (lebih menyeluruh)"
    )

# ============================================================================
# PROCESS BUTTON
# ============================================================================

st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    process_btn = st.button(
        "🚀 Proses Audio", 
        use_container_width=True,
        type="primary"
    )

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
    
    # Check file size
    if uploaded_file.size > 50 * 1024 * 1024:
        st.markdown("""
        <div class="error-box">
            ❌ File terlalu besar! Maksimal 50MB.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name
        
        # Create output path
        output_path = input_path.replace('.wav', '_processed.wav')
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("""
        <div class="info-box">
            ⏳ Memulai proses...
        </div>
        """, unsafe_allow_html=True)
        
        # Check if Python script exists
        script_path = Path(__file__).parent / "ai_audio_fingerprint_remover_enhanced.py"
        
        if not script_path.exists():
            st.markdown(f"""
            <div class="error-box">
                ❌ Script tidak ditemukan: {script_path}
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            input_path,
            output_path,
            "--level", level,
            "--tempo", str(tempo),
            "--pitch-semitones", str(pitch_semitones),
            "--pitch-percent", str(pitch_percent)
        ]
        
        if aggressive:
            cmd.append("--aggressive")
        
        # Update progress
        status_text.markdown("""
        <div class="info-box">
            🔄 Memproses audio... Ini mungkin memakan waktu 1-5 menit.
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(30)
        
        # Run the script
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update progress
        progress_bar.progress(90)
        
        # Check result
        if result.returncode != 0:
            st.markdown("""
            <div class="error-box">
                ❌ Gagal memproses audio.
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Detail Error"):
                st.code(result.stderr)
                st.code(result.stdout)
            st.stop()
        
        # Check if output exists
        if not os.path.exists(output_path):
            st.markdown("""
            <div class="error-box">
                ❌ File output tidak ditemukan.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Read output file
        with open(output_path, 'rb') as f:
            audio_data = f.read()
        
        # Parse stats from output
        stdout = result.stdout
        watermarks_removed = 0
        patterns_normalized = 0
        
        watermarks_match = re.search(r'Watermarks removed:\s*(\d+)', stdout, re.IGNORECASE)
        if watermarks_match:
            watermarks_removed = int(watermarks_match.group(1))
        
        patterns_match = re.search(r'Patterns normalized:\s*(\d+)', stdout, re.IGNORECASE)
        if patterns_match:
            patterns_normalized = int(patterns_match.group(1))
        
        # Cleanup temp files
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass
        
        # Complete progress
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
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
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
                <div class="stat-value">{patterns_normalized}</div>
                <div class="stat-label">Pattern Dinormalisasi</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size:1.2rem;">{level}</div>
                <div class="stat-label">Processing Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{processing_time:.1f}s</div>
                <div class="stat-label">Waktu Proses</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Download section
        st.markdown("### 📥 Download Hasil")
        
        # Create download button
        b64 = base64.b64encode(audio_data).decode()
        filename = f"processed_{uploaded_file.name}"
        href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}" class="download-btn">⬇️ Download Audio (WAV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Success message
        st.markdown("""
        <div class="success-box">
            ✅ Audio berhasil diproses! Watermark dan fingerprint AI telah dihapus.
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed log
        with st.expander("📋 Detail Log"):
            st.code(stdout)
            if result.stderr:
                st.warning("⚠️ Warning/Error output:")
                st.code(result.stderr)
        
    except subprocess.TimeoutExpired:
        st.markdown("""
        <div class="error-box">
            ⏰ Proses timeout (lebih dari 5 menit). Coba dengan file yang lebih kecil.
        </div>
        """, unsafe_allow_html=True)
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
    
    1. **Metadata Cleaning**: Menghapus tag AI dari metadata file
    2. **Watermark Detection**: Mendeteksi watermark di frekuensi tertentu
    3. **Watermark Removal**: Menghapus watermark menggunakan filter spectral
    4. **Pattern Normalization**: Menormalkan pola audio yang tidak natural
    5. **Timing Variations**: Menambahkan variasi timing alami
    
    ### Format yang Didukung
    
    | Format | Support |
    |--------|---------|
    | MP3 | ✅ Full |
    | WAV | ✅ Full |
    | FLAC | ✅ Full |
    | AIFF | ✅ Full |
    
    ### Keterbatasan
    
    - File maksimal 50MB (Streamlit Cloud limit)
    - Proses memakan waktu 1-5 menit tergantung ukuran file
    - Audio yang sangat pendek (< 5 detik) mungkin tidak terdeteksi watermark-nya
    """)

st.markdown("""
<div class="footer">
    Dibangun dengan ❤️ menggunakan Streamlit &bull; AI Audio Fingerprint Remover v3.0
    <br>
    <a href="#" target="_blank">GitHub</a> &bull; 
    <a href="#" target="_blank">Dokumentasi</a>
</div>
""", unsafe_allow_html=True)
