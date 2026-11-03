"""Pemodelan dengan LIBRARY (scikit-learn) - setara Modul 1A.

Studi kasus: prediksi kematian pasien gagal jantung.
Dataset: Heart Failure Clinical Records (Kaggle, 299 pasien).

Jalankan: python train_model.py
Hasil:
- data/dataset_split.csv   : dataset + kolom 'bagian' (train/test), dipakai juga oleh evaluasi LLM
- model.pkl                : model terbaik (dipakai aplikasi web)
- model_info.json          : fitur, metrik, versi library
- sampel_data_uji.csv      : contoh data untuk diunduh & diuji di web
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "heart_failure_clinical_records_dataset.csv"
TARGET = "DEATH_EVENT"
# Kolom 'time' (lama masa follow-up) TIDAK dipakai: nilainya baru diketahui
# setelah pasien dipantau, sehingga tidak tersedia saat prediksi (data leakage).
FITUR = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking",
]
NAMA_KELAS = {0: "Selamat", 1: "Meninggal"}
RANDOM_STATE = 42


def main():
    # 1. Muat dan periksa data
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"Missing values: {int(df.isna().sum().sum())} | Duplikat: {int(df.duplicated().sum())}")
    print("Distribusi kelas:")
    print(df[TARGET].map(NAMA_KELAS).value_counts().to_string())

    # 2. Bagi train/test (stratified, 80:20)
    train, test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE,
    )
    split = df.copy()
    split["bagian"] = "train"
    split.loc[test.index, "bagian"] = "test"
    split.insert(0, "id", range(1, len(split) + 1))
    split.to_csv(PROJECT_DIR / "data" / "dataset_split.csv", index=False)
    print(f"\nTrain: {len(train)} | Test: {len(test)}")

    X_train, y_train = train[FITUR], train[TARGET]
    X_test, y_test = test[FITUR], test[TARGET]

    # 3. Bandingkan beberapa metode dengan 5-fold cross-validation (hanya data train)
    kandidat = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    skor_cv = {}
    print("\nF1 macro cross-validation (data train):")
    for nama, model in kandidat.items():
        skor = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        skor_cv[nama] = round(float(skor.mean()), 4)
        print(f"- {nama}: {skor.mean():.4f} (+/- {skor.std():.4f})")

    nama_terbaik = max(skor_cv, key=skor_cv.get)
    model = kandidat[nama_terbaik]
    print(f"\nModel terpilih: {nama_terbaik}")

    # 4. Latih ulang pada seluruh data train, lalu evaluasi sekali pada data test
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    # Baseline: selalu menebak kelas mayoritas
    f1_baseline = f1_score(y_test, np.zeros_like(y_test), average="macro", zero_division=0)

    print("\nHASIL TEST")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1:.4f} (baseline selalu 'Selamat': {f1_baseline:.4f})")
    print(classification_report(
        y_test, pred, target_names=[NAMA_KELAS[0], NAMA_KELAS[1]], zero_division=0,
    ))
    print(pd.DataFrame(
        confusion_matrix(y_test, pred),
        index=[f"asli_{NAMA_KELAS[0]}", f"asli_{NAMA_KELAS[1]}"],
        columns=[f"pred_{NAMA_KELAS[0]}", f"pred_{NAMA_KELAS[1]}"],
    ))

    if nama_terbaik == "Random Forest":
        penting = pd.Series(model.feature_importances_, index=FITUR).sort_values(ascending=False)
        print("\nFitur paling berpengaruh:")
        print(penting.round(3).to_string())

    # 5. Simpan model dan informasinya
    joblib.dump(model, PROJECT_DIR / "model.pkl")
    info = {
        "nama_model": nama_terbaik,
        "fitur": FITUR,
        "target": TARGET,
        "kelas": NAMA_KELAS,
        "fitur_tidak_dipakai": {"time": "data leakage (diketahui setelah follow-up)"},
        "rentang_fitur": {
            kolom: [float(df[kolom].min()), float(df[kolom].max())] for kolom in FITUR
        },
        "cv_f1_macro": skor_cv,
        "metrik": {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1), 4),
            "f1_baseline": round(float(f1_baseline), 4),
            "n_test": int(len(test)),
        },
        "tanggal": datetime.now().strftime("%Y-%m-%d"),
        "versi": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    with open(PROJECT_DIR / "model_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False)

    # 6. Sampel data untuk diuji di web (5 per kelas dari data TEST)
    sampel = test.groupby(TARGET).sample(5, random_state=RANDOM_STATE)
    sampel[FITUR + [TARGET]].sample(frac=1, random_state=RANDOM_STATE).to_csv(
        PROJECT_DIR / "sampel_data_uji.csv", index=False,
    )

    print("\nTersimpan: model.pkl, model_info.json, data/dataset_split.csv, sampel_data_uji.csv")
    print(f"PENTING: versi scikit-learn {sklearn.__version__} harus sama di requirements.txt")


if __name__ == "__main__":
    main()
