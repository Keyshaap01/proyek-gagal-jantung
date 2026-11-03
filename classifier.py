"""Klasifikasi risiko kematian pasien gagal jantung melalui Groq (AI model service).

Fungsi yang SAMA dipakai saat evaluasi (evaluasi_llm.py) dan saat deploy (app.py),
sehingga perilaku model di web identik dengan saat dievaluasi.
"""

import json
import re
import time

import groq

# Nama kolom -> keterangan yang dibaca LLM
NAMA_FITUR = {
    "age": "Usia (tahun)",
    "anaemia": "Anemia",
    "creatinine_phosphokinase": "Enzim CPK (mcg/L)",
    "diabetes": "Diabetes",
    "ejection_fraction": "Ejection fraction (%)",
    "high_blood_pressure": "Hipertensi",
    "platelets": "Trombosit (per mL)",
    "serum_creatinine": "Kreatinin serum (mg/dL)",
    "serum_sodium": "Natrium serum (mEq/L)",
    "sex": "Jenis kelamin",
    "smoking": "Merokok",
}
FITUR_YA_TIDAK = {"anaemia", "diabetes", "high_blood_pressure", "smoking"}


def load_config(path="config.json"):
    """Baca konfigurasi yang telah lolos evaluasi."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def format_nilai(kolom, nilai):
    """Ubah nilai 0/1 menjadi teks yang mudah dipahami LLM."""
    if kolom in FITUR_YA_TIDAK:
        return "Ya" if int(float(nilai)) == 1 else "Tidak"
    if kolom == "sex":
        return "Laki-laki" if int(float(nilai)) == 1 else "Perempuan"
    angka = float(nilai)
    return f"{angka:g}"


def format_data(data):
    """Ubah dict data seorang pasien menjadi satu baris teks."""
    return "; ".join(
        f"{nama}: {format_nilai(kolom, data[kolom])}" for kolom, nama in NAMA_FITUR.items()
    )


def build_messages(cfg, data):
    """Susun instruksi, contoh berlabel (few-shot), lalu data yang akan diprediksi."""
    messages = [{"role": "system", "content": cfg["system_prompt"]}]
    for contoh in cfg.get("few_shot", []):
        messages.append({"role": "user", "content": format_data(contoh["data"])})
        messages.append({"role": "assistant", "content": contoh["label"]})
    messages.append({"role": "user", "content": format_data(data)})
    return messages


def parse_label(output, labels):
    """Rapikan label; jawaban kosong atau di luar daftar dihitung invalid."""
    jawaban = re.sub(r"[^a-z]", "", (output or "").lower())
    label_map = {label.lower(): label for label in labels}
    return label_map.get(jawaban, "invalid")


def classify(client, cfg, data, max_retry=3):
    """Panggil model sesuai konfigurasi, dengan retry saat terkena rate limit."""
    for attempt in range(max_retry):
        try:
            options = {}
            for key in ("reasoning_effort", "reasoning_format"):
                if key in cfg:
                    options[key] = cfg[key]
            response = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
                **options,
            )
            return parse_label(response.choices[0].message.content, cfg["labels"])
        except groq.RateLimitError:
            time.sleep(15 * (attempt + 1))
    return "invalid"
