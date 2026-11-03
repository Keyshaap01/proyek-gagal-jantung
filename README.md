# Prediksi Risiko Kematian Pasien Gagal Jantung

Studi kasus praktikum *Artificial Intelligence and Its Application*.
Dataset: [Heart Failure Clinical Records (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data), 299 pasien.

Dua pendekatan model, disajikan dalam **satu aplikasi web Streamlit**:

| Pendekatan | File | Setara modul |
|---|---|---|
| Library (scikit-learn) | `train_model.py` → `model.pkl` | 1A / 1B |
| AI model service (Groq) | `evaluasi_llm.py` → `config.json` | 2A / 2B |

## Urutan menjalankan (lokal, di VS Code)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Model library
python train_model.py

# 2. Model LLM (isi .env dulu: salin .env.example -> .env)
python evaluasi_llm.py --iterasi 1
python evaluasi_llm.py --iterasi 2
python evaluasi_llm.py --iterasi 3
python evaluasi_llm.py --test

# 3. Aplikasi web (isi .streamlit/secrets.toml dulu)
streamlit run app.py
```

## Struktur

```
proyek-gagal-jantung/
├── data/heart_failure_clinical_records_dataset.csv
├── train_model.py        # pemodelan library + evaluasi
├── classifier.py         # fungsi LLM (dipakai evaluasi & app)
├── evaluasi_llm.py       # iterasi prompt + gerbang evaluasi
├── app.py                # aplikasi web
├── requirements.txt
├── .env.example          # contoh API key untuk evaluasi lokal
└── .streamlit/secrets.toml.example
```

File yang dihasilkan: `model.pkl`, `model_info.json`, `config.json`, `sampel_data_uji.csv`,
`data/dataset_split.csv`.
