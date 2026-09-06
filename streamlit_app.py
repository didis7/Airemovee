import streamlit as st
import subprocess
import tempfile
import os
import sys
import hashlib
import json
from datetime import datetime

st.set_page_config(
    page_title="Audio Fingerprint Remover",
    page_icon="🎵",
    layout="wide"
)

# =====================================================================
# KONFIGURASI
# =====================================================================
SECRET_KEY = "RAHASIA2025"
DB_FILE = "licenses.json"

WA_NUMBER = "62881022005320"
WA_LINK = f"https://wa.me/{WA_NUMBER}?text=Saya%20mau%20beli%20akses%20Audio%20Remover"
TG_LINK = "https://t.me/didisugiant"

BANK_NAME = "BCA"
ACCOUNT_NUMBER = "0552596381"
ACCOUNT_NAME = "DIDI SUGIANTO"

# =====================================================================
# BAHASA
# =====================================================================
if "lang" not in st.session_state:
    st.session_state.lang = "id"  # id / en

def t(key):
    texts = {
        # === UMUM ===
        "app_title": {"id": "Audio Fingerprint Remover", "en": "Audio Fingerprint Remover"},
        "app_subtitle": {"id": "Hapus watermark AI dari file audio", "en": "Remove AI-generated watermarks from audio files"},
        "contact_support": {"id": "Butuh bantuan? Hubungi support@domain.com", "en": "Need help? Contact support@domain.com"},
        "copyright": {"id": "© Audio Fingerprint Remover 2025", "en": "© Audio Fingerprint Remover 2025"},
        "sign_out": {"id": "Keluar", "en": "Sign Out"},

        # === LOGIN ===
        "pricing": {"id": "💰 Harga", "en": "💰 Pricing"},
        "price": {"id": "Rp 15.000", "en": "Rp 15.000"},
        "lifetime": {"id": "Akses seumur hidup — bayar sekali", "en": "Lifetime access — one-time payment"},
        "what_you_get": {"id": "✅ Yang Anda Dapatkan", "en": "✅ What You Get"},
        "feature_unlimited": {"id": "Proses audio tanpa batas", "en": "Unlimited audio processing"},
        "feature_levels": {"id": "4 level pemrosesan (gentle, moderate, aggressive, extreme)", "en": "4 processing levels (gentle, moderate, aggressive, extreme)"},
        "feature_formats": {"id": "Dukung WAV, MP3, FLAC, AIFF", "en": "Support for WAV, MP3, FLAC, AIFF"},
        "feature_download": {"id": "Unduh hasil langsung", "en": "Direct download results"},
        "feature_lifetime": {"id": "Akses seumur hidup, tanpa langganan", "en": "Lifetime access, no subscription"},
        "payment_instructions": {"id": "🏦 Instruksi Pembayaran", "en": "🏦 Payment Instructions"},
        "bank_transfer": {"id": "Transfer Bank", "en": "Bank Transfer"},
        "payment_steps": {"id": """
        1. Transfer Rp 15.000 ke rekening di atas
        2. Kirim bukti transfer via WhatsApp atau Telegram
        3. Sertakan email Anda di pesan
        4. Terima kode akses dalam 30 menit
        """, "en": """
        1. Transfer Rp 15.000 to the account above
        2. Send proof of payment via WhatsApp or Telegram
        3. Include your email in the message
        4. Receive access code within 30 minutes
        """},
        "contact_us": {"id": "📱 Hubungi Kami", "en": "📱 Contact Us"},
        "enter_access_code": {"id": "🔑 Masukkan Kode Akses", "en": "🔑 Enter Access Code"},
        "already_purchased": {"id": "Sudah membeli? Masukkan kode Anda di bawah.", "en": "Already purchased? Enter your code below."},
        "email": {"id": "Email", "en": "Email"},
        "access_code": {"id": "Kode Akses", "en": "Access Code"},
        "verify_access": {"id": "Verifikasi Akses", "en": "Verify Access"},
        "fill_both": {"id": "Isi email dan kode akses", "en": "Please fill in both email and access code"},
        "invalid_credentials": {"id": "Email atau kode akses salah", "en": "Invalid email or access code"},
        "forgot_code": {"id": "Hubungi support jika lupa kode", "en": "Contact support if you forgot your code"},
        "try_before_buy": {"id": "🧪 Ingin coba sebelum beli?", "en": "🧪 Need to try before buying?"},
        "trial_contact": {"id": "Hubungi kami untuk akses trial.", "en": "Contact us for a trial access."},

        # === SIDEBAR ===
        "account": {"id": "👤 Akun", "en": "👤 Account"},
        "license_active": {"id": "✅ Lisensi: Aktif", "en": "✅ License: Active"},
        "license_trial": {"id": "⚠️ Lisensi: Trial", "en": "⚠️ License: Trial"},
        "audio_effects": {"id": "🎚️ Efek Audio", "en": "🎚️ Audio Effects"},
        "tempo_change": {"id": "Perubahan Tempo (%)", "en": "Tempo Change (%)"},
        "tempo_help": {"id": "Negatif = lebih lambat, Positif = lebih cepat", "en": "Negative = slower, Positive = faster"},
        "pitch_shift": {"id": "Perubahan Nada (semitones)", "en": "Pitch Shift (semitones)"},
        "pitch_help": {"id": "Negatif = nada lebih rendah, Positif = nada lebih tinggi", "en": "Negative = lower pitch, Positive = higher pitch"},
        "pitch_percent": {"id": "Perubahan Nada (%)", "en": "Pitch Shift (%)"},
        "pitch_percent_help": {"id": "Penyesuaian nada berbasis persentase", "en": "Percentage-based pitch adjustment"},
        "tempo_pitch_label": {"id": "Tempo: {}% | Nada: {} semitones / {}%", "en": "Tempo: {}% | Pitch: {} semitones / {}%"},
        "processing_level": {"id": "Level Pemrosesan", "en": "Processing Level"},
        "level_help": {"id": "Gentle: menjaga kualitas | Extreme: penghapusan maksimal", "en": "Gentle: preserves quality | Extreme: maximum removal"},

        # === UPLOAD ===
        "upload_audio": {"id": "📂 Unggah File Audio", "en": "📂 Upload Audio File"},
        "upload_help": {"id": "Format didukung: WAV, MP3, FLAC, AIFF", "en": "Supported formats: WAV, MP3, FLAC, AIFF"},
        "drop_here": {"id": "📂 Taruh file audio di sini atau klik untuk cari", "en": "📂 Drop your audio file here or click to browse"},
        "file_name": {"id": "📄 Nama File", "en": "📄 File Name"},
        "file_size": {"id": "📦 Ukuran", "en": "📦 Size"},
        "process_level": {"id": "⚙️ Level", "en": "⚙️ Level"},
        "process_audio": {"id": "🚀 Proses Audio", "en": "🚀 Process Audio"},
        "processing": {"id": "⏳ Memproses audio...", "en": "⏳ Processing audio..."},
        "file_not_found": {"id": "❌ File tidak ditemukan: {}", "en": "❌ File not found: {}"},
        "process_failed": {"id": "❌ Proses gagal dengan kode {}", "en": "❌ Process failed with code {}"},
        "process_success": {"id": "✅ Pemrosesan selesai", "en": "✅ Processing completed"},
        "original": {"id": "🎧 Original", "en": "🎧 Original"},
        "processed": {"id": "🎛️ Hasil Proses", "en": "🎛️ Processed"},
        "download_result": {"id": "⬇️ Unduh Hasil", "en": "⬇️ Download Result"},
        "output_not_created": {"id": "❌ Pemrosesan gagal. File output tidak dibuat.", "en": "❌ Processing failed. Output file not created."},
        "timeout": {"id": "⏰ Pemrosesan habis waktu (300 detik)", "en": "⏰ Processing timed out (300 seconds)"},
        "error": {"id": "⚠️ Error: {}", "en": "⚠️ Error: {}"},
    }
    return texts.get(key, {}).get(st.session_state.lang, key)

# =====================================================================
# SESSION STATE
# =====================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_email = ""

# =====================================================================
# FUNGSI LISENSI
# =====================================================================
def load_licenses():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_licenses(licenses):
    with open(DB_FILE, "w") as f:
        json.dump(licenses, f, indent=2)

def generate_license_key(email):
    combined = email + SECRET_KEY
    return hashlib.sha256(combined.encode()).hexdigest()[:16].upper()

def validate_license(email, key):
    if email == "demo@coba.com":
        return False
    expected = generate_license_key(email)
    clean_key = key.replace("-", "").upper()
    return clean_key == expected

# =====================================================================
# TOMBOL BAHASA
# =====================================================================
col_lang, _ = st.columns([1, 5])
with col_lang:
    lang_options = {"🇮🇩 Indonesia": "id", "🇬🇧 English": "en"}
    current_label = [k for k, v in lang_options.items() if v == st.session_state.lang][0]
    selected = st.selectbox("🌐", list(lang_options.keys()), index=list(lang_options.values()).index(st.session_state.lang), label_visibility="collapsed")
    if lang_options[selected] != st.session_state.lang:
        st.session_state.lang = lang_options[selected]
        st.rerun()

# =====================================================================
# HALAMAN LOGIN
# =====================================================================
if not st.session_state.authenticated:
    st.markdown(f"# 🎵 {t('app_title')}")
    st.markdown(f"*{t('app_subtitle')}*")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"### {t('pricing')}")
        st.markdown(f"## **{t('price')}**")
        st.caption(t('lifetime'))
        st.divider()

        st.markdown(f"### {t('what_you_get')}")
        st.markdown(f"""
        - {t('feature_unlimited')}
        - {t('feature_levels')}
        - {t('feature_formats')}
        - {t('feature_download')}
        - {t('feature_lifetime')}
        """)
        st.divider()

        st.markdown(f"### {t('payment_instructions')}")
        st.markdown(f"**{t('bank_transfer')}**")
        st.code(f"{BANK_NAME} - {ACCOUNT_NUMBER} - {ACCOUNT_NAME}")

        st.markdown(t('payment_steps'))
        st.divider()

        st.markdown(f"### {t('contact_us')}")
        col_wa, col_tg = st.columns(2)
        with col_wa:
            st.link_button("💬 WhatsApp", WA_LINK, type="primary", use_container_width=True)
        with col_tg:
            st.link_button("✈️ Telegram", TG_LINK, use_container_width=True)

    with col2:
        st.markdown(f"### {t('enter_access_code')}")
        st.caption(t('already_purchased'))

        with st.form("login_form"):
            email = st.text_input(t('email'), placeholder="your@email.com")
            license_key = st.text_input(t('access_code'), type="password", placeholder="XXXX-XXXX-XXXX-XXXX")
            submitted = st.form_submit_button(t('verify_access'), type="primary", use_container_width=True)

            if submitted:
                if not email or not license_key:
                    st.error(t('fill_both'))
                else:
                    if validate_license(email, license_key):
                        licenses = load_licenses()
                        if email in licenses:
                            st.session_state.authenticated = True
                            st.session_state.user_email = email
                            st.rerun()
                        else:
                            licenses[email] = {
                                "key": generate_license_key(email),
                                "activated_at": datetime.now().isoformat(),
                                "status": "active"
                            }
                            save_licenses(licenses)
                            st.session_state.authenticated = True
                            st.session_state.user_email = email
                            st.rerun()
                    else:
                        st.error(t('invalid_credentials'))
                        st.info(t('forgot_code'))

        st.divider()
        st.markdown(f"### {t('try_before_buy')}")
        st.markdown(t('trial_contact'))
        col_wa2, col_tg2 = st.columns(2)
        with col_wa2:
            st.link_button("💬 WhatsApp", WA_LINK, use_container_width=True)
        with col_tg2:
            st.link_button("✈️ Telegram", TG_LINK, use_container_width=True)

    st.divider()
    st.caption(f"📧 {t('contact_support')}")
    st.stop()

# =====================================================================
# HALAMAN UTAMA
# =====================================================================
st.markdown(f"# 🎵 {t('app_title')}")
st.markdown(f"*{t('app_subtitle')}*")
st.divider()

with st.sidebar:
    st.markdown(f"### {t('account')}")
    st.markdown(f"**{st.session_state.user_email}**")
    licenses = load_licenses()
    if st.session_state.user_email in licenses:
        st.success(t('license_active'))
    else:
        st.warning(t('license_trial'))
    st.divider()

    st.markdown(f"### {t('audio_effects')}")

    tempo_initial = st.slider(
        t('tempo_change'),
        min_value=-50,
        max_value=50,
        value=0,
        step=1,
        help=t('tempo_help')
    )

    pitch_semitones = st.slider(
        t('pitch_shift'),
        min_value=-12,
        max_value=12,
        value=0,
        step=1,
        help=t('pitch_help')
    )

    pitch_percent = st.slider(
        t('pitch_percent'),
        min_value=-50,
        max_value=100,
        value=0,
        step=1,
        help=t('pitch_percent_help')
    )

    st.caption(t('tempo_pitch_label').format(tempo_initial, pitch_semitones, pitch_percent))
    st.divider()

    level = st.selectbox(
        t('processing_level'),
        ["gentle", "moderate", "aggressive", "extreme"],
        help=t('level_help')
    )
    st.divider()

    if st.button(f"🚪 {t('sign_out')}", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.rerun()

# =====================================================================
# UPLOAD & PROSES
# =====================================================================
uploaded_file = st.file_uploader(
    t('upload_audio'),
    type=["wav", "mp3", "flac", "aiff"],
    help=t('upload_help')
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = input_path + "_output.wav"

    col1, col2, col3 = st.columns(3)
    col1.metric(t('file_name'), uploaded_file.name)
    col2.metric(t('file_size'), f"{uploaded_file.size / 1024:.1f} KB")
    col3.metric(t('process_level'), level.capitalize())

    if st.button(t('process_audio'), type="primary"):
        with st.spinner(t('processing')):
            try:
                script_path = "ai_audio_fingerprint_remover_enhanced.py"

                if not os.path.exists(script_path):
                    st.error(t('file_not_found').format(script_path))
                    st.stop()

                cmd = [
                    sys.executable, script_path,
                    input_path, output_path,
                    "--level", level,
                    "--no-ml",
                    "--tempo", str(tempo_initial),
                    "--pitch-semitones", str(pitch_semitones),
                    "--pitch-percent", str(pitch_percent)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                def filter_warnings(text):
                    lines = text.split("\n")
                    filtered = [
                        line for line in lines
                        if not any(x in line for x in ["FutureWarning", "deprecated", "librosa/effects.py"])
                    ]
                    return "\n".join(filtered)

                stdout_clean = filter_warnings(result.stdout)
                stderr_clean = filter_warnings(result.stderr)

                if stdout_clean.strip():
                    st.text("Output:")
                    st.code(stdout_clean)
                if stderr_clean.strip():
                    st.text("Error Output:")
                    st.code(stderr_clean)

                if result.returncode != 0:
                    st.error(t('process_failed').format(result.returncode))
                    st.stop()

                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    st.success(t('process_success'))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(t('original'))
                        st.audio(input_path)
                    with col2:
                        st.subheader(t('processed'))
                        st.audio(output_path)

                    with open(output_path, "rb") as f:
                        st.download_button(
                            t('download_result'),
                            f,
                            file_name="processed_" + uploaded_file.name,
                            mime="audio/wav"
                        )
                else:
                    st.error(t('output_not_created'))

            except subprocess.TimeoutExpired:
                st.error(t('timeout'))
            except Exception as e:
                st.error(t('error').format(e))
                import traceback
                st.code(traceback.format_exc())

            finally:
                try:
                    if os.path.exists(input_path):
                        os.unlink(input_path)
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except:
                    pass

else:
    st.info(t('drop_here'))
    st.caption(t('upload_help'))

st.divider()
st.caption(t('copyright'))
