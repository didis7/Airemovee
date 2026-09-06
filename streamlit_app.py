import streamlit as st
import os
import hashlib
import json
from datetime import datetime

# =====================================================================
# KONFIGURASI
# =====================================================================
st.set_page_config(
    page_title="Audio Fingerprint Remover",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# KONFIGURASI
# =====================================================================
SECRET_KEY = "sugianti123@"
DB_FILE = "licenses.json"

WA_NUMBER = "62881022005320"
WA_LINK = f"https://wa.me/{WA_NUMBER}?text=Saya%20mau%20beli%20akses%20Audio%20Remover"
TG_LINK = "https://t.me/didisugiant"

BANK_NAME = "BCA"
ACCOUNT_NUMBER = "0552596381"
ACCOUNT_NAME = "DIDI SUGIANTO"

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
# SESSION STATE
# =====================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_email = ""

# =====================================================================
# HALAMAN LOGIN
# =====================================================================
if not st.session_state.authenticated:
    st.markdown("# 🎵 Audio Fingerprint Remover")
    st.markdown("*Hapus watermark AI dari file audio*")
    st.divider()

    with st.container():
        st.markdown("### 💰 Harga")
        st.markdown("## **Rp 15.000**")
        st.caption("Akses seumur hidup — bayar sekali")
        st.divider()

        st.markdown("### ✅ Yang Didapat")
        st.markdown("""
        - Proses audio tanpa batas
        - 4 level pemrosesan
        - WAV, MP3, FLAC, AIFF
        - Unduh hasil langsung
        - Akses seumur hidup
        """)
        st.divider()

        st.markdown("### 🏦 Bayar ke")
        st.code(f"{BANK_NAME} - {ACCOUNT_NUMBER} - {ACCOUNT_NAME}")
        st.markdown("""
        1. Transfer Rp 15.000
        2. Kirim bukti ke WhatsApp
        3. Sertakan email
        4. Dapatkan kode akses
        """)
        st.divider()

        st.markdown("### 🔑 Masukkan Kode Akses")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="email@domain.com")
            license_key = st.text_input("Kode Akses", type="password", placeholder="XXXX-XXXX-XXXX-XXXX")
            submitted = st.form_submit_button("Verifikasi", type="primary", use_container_width=True)

            if submitted:
                if not email or not license_key:
                    st.error("Isi email dan kode akses")
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
                        st.error("Email atau kode akses salah")

        st.divider()
        col_wa, col_tg = st.columns(2)
        with col_wa:
            st.link_button("💬 WhatsApp", WA_LINK, type="primary", use_container_width=True)
        with col_tg:
            st.link_button("✈️ Telegram", TG_LINK, use_container_width=True)

    st.caption("📧 Butuh bantuan? Hubungi support@domain.com")
    st.stop()

# =====================================================================
# HALAMAN UTAMA
# =====================================================================
st.markdown("# 🎵 Audio Fingerprint Remover")
st.divider()

with st.sidebar:
    st.markdown(f"**👤 {st.session_state.user_email}**")
    licenses = load_licenses()
    if st.session_state.user_email in licenses:
        st.success("✅ Lisensi Aktif")
    else:
        st.warning("⚠️ Trial")
    st.divider()

    st.markdown("### 🎚️ Efek Audio")
    tempo = st.slider("Tempo (%)", -50, 50, 0)
    pitch = st.slider("Nada (semitones)", -12, 12, 0)
    level = st.selectbox("Level", ["gentle", "moderate", "aggressive", "extreme"])
    st.divider()

    if st.button("🚪 Keluar", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.rerun()

# =====================================================================
# UPLOAD
# =====================================================================
st.markdown("### 📥 Upload Audio")

uploaded_file = st.file_uploader(
    "Pilih file audio dari HP",
    type=["wav", "mp3", "flac", "aiff", "m4a", "ogg"],
    help="Maks 50MB"
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > 50:
        st.error(f"❌ File terlalu besar! {file_size_mb:.1f} MB > 50 MB")
        st.stop()
    
    st.success(f"✅ {uploaded_file.name} ({file_size_mb:.1f} MB)")
    
    st.markdown("---")
    st.markdown("### 🚀 Proses Audio")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 File", uploaded_file.name[:25] + ("..." if len(uploaded_file.name) > 25 else ""))
    with col2:
        st.metric("📦 Ukuran", f"{file_size_mb:.1f} MB")
    
    # INFO: Proses audio coming soon
    st.info("""
    ⚠️ **Mode Demo**
    
    Fitur proses audio sedang dalam pengembangan.
    Saat ini Anda bisa upload file dan atur efek.
    Proses audio akan segera tersedia!
    """)
    
    if st.button("🎵 Proses Sekarang (Coming Soon)", type="primary", use_container_width=True, disabled=True):
        st.info("Fitur ini akan segera tersedia!")

else:
    st.info("📂 Klik tombol di atas untuk pilih file dari HP")
    st.caption("Support: WAV, MP3, FLAC, AIFF, M4A, OGG — Maks 50MB")

st.divider()
st.caption("© Audio Fingerprint Remover 2025")
