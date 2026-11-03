"""Aplikasi web prediksi risiko kematian pasien gagal jantung.

Satu aplikasi, dua model:
1. Model library  : scikit-learn (model.pkl hasil train_model.py)
2. Model LLM      : Groq (config.json hasil evaluasi_llm.py yang lolos gerbang)
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from classifier import NAMA_FITUR, classify, load_config

DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Prediksi Gagal Jantung", page_icon="🫀", layout="wide")


# ---------- 1. Muat model library ----------
@st.cache_resource
def muat_model_library():
    model = joblib.load(DIR / "model.pkl")
    with open(DIR / "model_info.json", encoding="utf-8") as file:
        info = json.load(file)
    return model, info


model_lib, info_lib = muat_model_library()
FITUR = info_lib["fitur"]
KELAS = {int(k): v for k, v in info_lib["kelas"].items()}  # {0: Selamat, 1: Meninggal}
RENTANG = info_lib["rentang_fitur"]

# ---------- 2. Muat model LLM (opsional: aplikasi tetap jalan jika belum siap) ----------
cfg_llm, client_llm, pesan_llm = None, None, ""
if (DIR / "config.json").exists():
    cfg_llm = load_config(DIR / "config.json")
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:  # file secrets belum ada atau key belum diisi
        api_key = None
    if api_key:
        from groq import Groq

        @st.cache_resource
        def get_client(key):
            return Groq(api_key=key)

        client_llm = get_client(api_key)
    else:
        pesan_llm = ("GROQ_API_KEY belum diatur. Isi di .streamlit/secrets.toml (lokal) "
                     "atau menu Secrets (Streamlit Community Cloud).")
else:
    pesan_llm = "config.json belum ada. Jalankan evaluasi_llm.py sampai lolos gerbang evaluasi."
LLM_SIAP = client_llm is not None


@st.cache_data(show_spinner=False)
def prediksi_llm(data_tuple):
    # Data yang sama tidak dikirim dua kali ke API (hemat kuota)
    return classify(client_llm, cfg_llm, dict(data_tuple))


def prediksi_library(df):
    proba = model_lib.predict_proba(df[FITUR])[:, 1]
    label = [KELAS[int(p >= 0.5)] for p in proba]
    return label, proba


def tampilkan_label(label, keterangan=""):
    if label == "Meninggal":
        st.error(f"### Risiko TINGGI: **{label}** {keterangan}")
    elif label == "Selamat":
        st.success(f"### Risiko RENDAH: **{label}** {keterangan}")
    else:
        st.warning(f"Model tidak memberi jawaban yang valid ({label}). Coba lagi.")


def form_input(key):
    """Form input data pasien. Mengembalikan dict (format sama dengan dataset) atau None."""
    with st.form(f"form_{key}"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Usia (tahun)", 18, 100, 60, key=f"{key}_age")
            sex = st.radio("Jenis kelamin", ["Laki-laki", "Perempuan"], horizontal=True,
                           key=f"{key}_sex")
            smoking = st.radio("Merokok?", ["Tidak", "Ya"], horizontal=True, key=f"{key}_smk")
            diabetes = st.radio("Diabetes?", ["Tidak", "Ya"], horizontal=True, key=f"{key}_dm")
        with c2:
            ef = st.slider("Ejection fraction (%)", 10, 80, 38, key=f"{key}_ef",
                           help="Persentase darah yang dipompa jantung tiap detak. Normal 50-70%.")
            kreatinin = st.number_input("Kreatinin serum (mg/dL)", 0.3, 10.0, 1.1, 0.1,
                                        key=f"{key}_cr", help="Fungsi ginjal. Normal 0,6-1,2.")
            natrium = st.number_input("Natrium serum (mEq/L)", 110, 150, 137, key=f"{key}_na",
                                      help="Normal 135-145.")
            hipertensi = st.radio("Hipertensi?", ["Tidak", "Ya"], horizontal=True,
                                  key=f"{key}_hbp")
        with c3:
            cpk = st.number_input("Enzim CPK (mcg/L)", 10, 8000, 250, key=f"{key}_cpk")
            trombosit = st.number_input("Trombosit (per mL)", 20000, 900000, 262000, 1000,
                                        key=f"{key}_plt")
            anemia = st.radio("Anemia?", ["Tidak", "Ya"], horizontal=True, key=f"{key}_ana")
        kirim = st.form_submit_button("Prediksi", type="primary")
    if not kirim:
        return None
    return {
        "age": float(age),
        "anaemia": int(anemia == "Ya"),
        "creatinine_phosphokinase": int(cpk),
        "diabetes": int(diabetes == "Ya"),
        "ejection_fraction": int(ef),
        "high_blood_pressure": int(hipertensi == "Ya"),
        "platelets": float(trombosit),
        "serum_creatinine": float(kreatinin),
        "serum_sodium": int(natrium),
        "sex": int(sex == "Laki-laki"),
        "smoking": int(smoking == "Ya"),
    }


# ---------- 3. Sidebar ----------
with st.sidebar:
    st.header("🫀 Tentang Aplikasi")
    st.write("Prediksi risiko kematian pasien **gagal jantung** dari 11 data klinis, "
             "dengan dua pendekatan model.")
    st.subheader("Model library")
    st.write(f"`{info_lib['nama_model']}` (scikit-learn)")
    st.metric("F1 macro (data test)", f"{info_lib['metrik']['f1_macro']:.3f}")
    st.subheader("Model LLM")
    if cfg_llm:
        ev = cfg_llm["evaluasi"]
        st.write(f"`{cfg_llm['model']}` (Groq), {ev['nama_iterasi']}")
        st.metric("F1 macro (data test)", f"{ev['f1_macro']:.3f}")
        with st.expander("Lihat system prompt"):
            st.code(cfg_llm["system_prompt"], language=None)
    else:
        st.info("Belum tersedia.")
    st.caption(f"Kedua model dievaluasi pada {info_lib['metrik']['n_test']} data test yang sama.")
    st.divider()
    st.download_button(
        "⬇️ Unduh sampel data uji (CSV)",
        (DIR / "sampel_data_uji.csv").read_bytes(),
        file_name="sampel_data_uji.csv", mime="text/csv",
    )

st.title("🫀 Prediksi Risiko Kematian Pasien Gagal Jantung")
st.caption("Dataset: Heart Failure Clinical Records (Kaggle), 299 pasien. "
           "Aplikasi ini untuk keperluan praktikum, bukan pengganti diagnosis dokter.")

tab_lib, tab_llm, tab_csv, tab_data = st.tabs([
    "🌲 Model Library", "🤖 Model LLM (Groq)", "📄 Prediksi dari CSV", "📊 Tentang Data",
])

# ---------- 4. Tab model library ----------
with tab_lib:
    st.write(f"Prediksi dengan **{info_lib['nama_model']}** yang dilatih memakai scikit-learn.")
    data = form_input("lib")
    if data:
        label, proba = prediksi_library(pd.DataFrame([data]))
        tampilkan_label(label[0])
        c1, c2 = st.columns(2)
        c1.metric("Probabilitas Meninggal", f"{proba[0]:.1%}")
        c2.metric("Probabilitas Selamat", f"{1 - proba[0]:.1%}")

# ---------- 5. Tab model LLM ----------
with tab_llm:
    st.write("Prediksi dengan **LLM melalui Groq API**, memakai prompt yang sudah lolos evaluasi.")
    if not LLM_SIAP:
        st.warning(pesan_llm)
    data = form_input("llm")
    if data and LLM_SIAP:
        with st.spinner("Menghubungi Groq..."):
            label = prediksi_llm(tuple(sorted(data.items())))
        tampilkan_label(label)
        st.caption("LLM hanya mengembalikan label (tanpa probabilitas).")

# ---------- 6. Tab CSV ----------
with tab_csv:
    st.write("Unggah CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR) + ".")
    st.download_button(
        "⬇️ Unduh sampel data uji", (DIR / "sampel_data_uji.csv").read_bytes(),
        file_name="sampel_data_uji.csv", mime="text/csv", key="dl_csv_tab",
    )
    pilihan = st.radio("Model", ["Model Library", "Model LLM (maks. 30 baris)"],
                       horizontal=True)
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        elif pilihan == "Model Library":
            label, proba = prediksi_library(df)
            df["prediksi_library"] = label
            df["prob_meninggal"] = proba.round(3)
        elif not LLM_SIAP:
            st.warning(pesan_llm)
            df = None
        else:
            df = df.head(30)
            if st.button("Prediksi semua dengan LLM"):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, baris) in enumerate(df.iterrows(), start=1):
                    hasil.append(prediksi_llm(tuple(sorted(baris[FITUR].to_dict().items()))))
                    bar.progress(i / len(df))
                    time.sleep(2)  # jaga batas request per menit
                df["prediksi_llm"] = hasil
            else:
                df = None

        if df is not None and not kurang:
            kolom_pred = "prediksi_library" if "prediksi_library" in df else "prediksi_llm"
            st.dataframe(df)
            st.write(df[kolom_pred].value_counts())
            if "DEATH_EVENT" in df:
                benar = (df[kolom_pred] == df["DEATH_EVENT"].map(KELAS)).mean()
                st.info(f"Kolom DEATH_EVENT ditemukan: prediksi sesuai label asli pada "
                        f"{benar:.0%} baris.")
            st.download_button("⬇️ Unduh hasil prediksi", df.to_csv(index=False),
                               file_name="hasil_prediksi.csv", mime="text/csv")

# ---------- 7. Tab tentang data ----------
with tab_data:
    st.subheader("Fitur yang dipakai")
    st.table(pd.DataFrame({
        "Kolom": FITUR,
        "Keterangan": [NAMA_FITUR[c] for c in FITUR],
        "Rentang di dataset": [f"{RENTANG[c][0]:g} - {RENTANG[c][1]:g}" for c in FITUR],
    }))
    st.write("Kolom `time` (lama masa pemantauan) tidak dipakai karena nilainya baru "
             "diketahui setelah pasien dipantau, sehingga tidak tersedia saat prediksi.")
    st.subheader("Perbandingan model (data test)")
    perbandingan = {
        f"Library ({info_lib['nama_model']})": {
            "Accuracy": info_lib["metrik"]["accuracy"],
            "F1 macro": info_lib["metrik"]["f1_macro"],
        },
    }
    if cfg_llm:
        perbandingan["LLM (Groq)"] = {
            "Accuracy": cfg_llm["evaluasi"]["accuracy"],
            "F1 macro": cfg_llm["evaluasi"]["f1_macro"],
        }
    st.table(pd.DataFrame(perbandingan))
