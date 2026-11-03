"""Pemodelan dengan AI MODEL SERVICE (Groq) - setara Modul 2A.

Jalankan berurutan:
    python evaluasi_llm.py --iterasi 1   # zero-shot
    python evaluasi_llm.py --iterasi 2   # few-shot
    python evaluasi_llm.py --iterasi 3   # few-shot + aturan klinis dari data train
    python evaluasi_llm.py --test        # pilih iterasi terbaik, uji di data test, cek gerbang

Jika lolos gerbang evaluasi, config.json dibuat dan siap dipakai app.py.
Butuh: train_model.py sudah dijalankan (data/dataset_split.csv & model_info.json)
dan file .env berisi GROQ_API_KEY=gsk_...
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from classifier import NAMA_FITUR, classify

PROJECT_DIR = Path(__file__).resolve().parent
FITUR = list(NAMA_FITUR)
TARGET = "DEATH_EVENT"
LABELS = ["Selamat", "Meninggal"]
# Gerbang evaluasi. Target lebih rendah dari contoh modul (0,70) karena kasus ini
# lebih sulit: model library pun hanya mencapai F1 sekitar 0,65-0,74 tanpa kolom 'time'.
TARGET_F1 = 0.60
MAKS_INVALID_RATE = 0.05
JEDA = 1  # detik antar-request, menjaga batas request per menit

PROMPT_DASAR = (
    "Kamu adalah sistem pendukung keputusan klinis untuk pasien gagal jantung.\n"
    "Data pasien: usia, anemia, enzim CPK, diabetes, ejection fraction (persentase "
    "darah yang dipompa jantung setiap detak), hipertensi, trombosit, kreatinin serum "
    "(fungsi ginjal), natrium serum, jenis kelamin, dan kebiasaan merokok.\n"
    "Tugas: prediksi apakah pasien akan meninggal selama masa pemantauan.\n"
    "Label:\n"
    "- Meninggal: risiko kematian tinggi\n"
    "- Selamat: risiko kematian rendah\n"
    "Jawab HANYA dengan satu kata: Selamat atau Meninggal."
)

CFG_DASAR = {
    # Model yang tersedia di akun Groq; ganti jika model sudah tidak tersedia.
    "model": "qwen/qwen3.8-27b",
    "temperature": 0,
    "max_tokens": 16,
    "reasoning_effort": "none",
    "labels": LABELS,
    "system_prompt": PROMPT_DASAR,
    "few_shot": [],
}

NAMA_ITERASI = {1: "zero-shot", 2: "few-shot", 3: "few-shot + aturan klinis"}


def muat_data():
    path = PROJECT_DIR / "data" / "dataset_split.csv"
    if not path.exists():
        raise SystemExit("data/dataset_split.csv belum ada. Jalankan dulu: python train_model.py")
    df = pd.read_csv(path)
    df["status"] = df[TARGET].map({0: "Selamat", 1: "Meninggal"})
    train = df[df["bagian"] == "train"]
    test = df[df["bagian"] == "test"]
    # Validasi: 15 pasien per kelas dari TRAIN, untuk membandingkan prompt
    val = train.groupby("status").sample(15, random_state=7)
    pool = train.drop(val.index)  # sumber contoh few-shot & aturan klinis
    return train, val, pool, test


def evaluasi(client, cfg, data):
    """Klasifikasikan data dan kembalikan metrik serta prediksinya."""
    prediksi = []
    for i, (_, baris) in enumerate(data.iterrows(), start=1):
        prediksi.append(classify(client, cfg, baris[FITUR].to_dict()))
        print(f"\r{i}/{len(data)}", end="", flush=True)
        time.sleep(JEDA)
    print()

    y_true = data["status"].tolist()
    hasil = {
        "accuracy": round(accuracy_score(y_true, prediksi), 4),
        "f1_macro": round(f1_score(
            y_true, prediksi, labels=LABELS, average="macro", zero_division=0,
        ), 4),
        "invalid": prediksi.count("invalid"),
        "n": len(data),
    }
    print(hasil)
    print(classification_report(y_true, prediksi, labels=LABELS, zero_division=0))
    matrix = confusion_matrix(y_true, prediksi, labels=LABELS + ["invalid"])
    print(pd.DataFrame(
        matrix[:len(LABELS), :],
        index=[f"asli_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS + ["invalid"]],
    ))
    return hasil, prediksi


def cari_ambang(pool):
    """Cari ambang 'EF <= a ATAU kreatinin >= b' terbaik, HANYA dari data pool (train)."""
    terbaik = (0, None, None)
    for ef in range(20, 41, 5):
        for kreatinin in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
            pred = (pool["ejection_fraction"] <= ef) | (pool["serum_creatinine"] >= kreatinin)
            pred = pred.map({True: "Meninggal", False: "Selamat"})
            skor = f1_score(pool["status"], pred, average="macro")
            if skor > terbaik[0]:
                terbaik = (skor, ef, kreatinin)
    return terbaik


def buat_config(iterasi, pool):
    cfg = dict(CFG_DASAR)
    if iterasi >= 2:
        contoh = pool.groupby("status").sample(3, random_state=1)
        cfg["few_shot"] = [
            {"data": baris[FITUR].to_dict(), "label": baris["status"]}
            for _, baris in contoh.sample(frac=1, random_state=1).iterrows()
        ]
        print(f"\n{len(cfg['few_shot'])} contoh few-shot dari data train:")
        print(contoh[["id", *FITUR, "status"]].to_string(index=False))
    if iterasi >= 3:
        skor, ef, kreatinin = cari_ambang(pool)
        print(f"\nAturan dari data train: EF <= {ef}% ATAU kreatinin >= {kreatinin} "
              f"(F1 di data train: {skor:.3f})")
        cfg["system_prompt"] = (
            PROMPT_DASAR
            + "\n\nPedoman dari data historis (dua indikator terkuat):\n"
            f"- Ejection fraction {ef}% atau lebih rendah -> Meninggal\n"
            f"- Kreatinin serum {kreatinin} mg/dL atau lebih tinggi -> Meninggal\n"
            "- Jika KEDUA indikator normal -> Selamat\n"
            "Cukup bandingkan kedua angka tersebut dengan batasnya.\n"
            "Jawab tepat satu kata: Selamat atau Meninggal. Tanpa penjelasan."
        )
    return cfg


def finalisasi(hasil_test, cfg, iterasi):
    """Bandingkan dengan model library dan cek gerbang evaluasi."""
    hasil = dict(hasil_test)
    info_path = PROJECT_DIR / "model_info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as file:
            lib = json.load(file)
        hasil["f1_library_pembanding"] = lib["metrik"]["f1_macro"]
        print("\nPERBANDINGAN DI DATA TEST YANG SAMA")
        print(pd.DataFrame({
            f"Library ({lib['nama_model']})": {
                "accuracy": lib["metrik"]["accuracy"], "f1_macro": lib["metrik"]["f1_macro"],
            },
            "LLM (Groq)": {"accuracy": hasil["accuracy"], "f1_macro": hasil["f1_macro"]},
        }).to_string())

    maks_invalid = MAKS_INVALID_RATE * hasil["n"]
    lolos_f1 = hasil["f1_macro"] >= TARGET_F1
    lolos_invalid = hasil["invalid"] <= maks_invalid
    print("\nGERBANG EVALUASI")
    print(f"F1 macro: {hasil['f1_macro']:.4f} | Target >= {TARGET_F1:.2f} | "
          f"{'LOLOS' if lolos_f1 else 'BELUM LOLOS'}")
    print(f"Invalid: {hasil['invalid']}/{hasil['n']} | Maksimal {maks_invalid:g} | "
          f"{'LOLOS' if lolos_invalid else 'BELUM LOLOS'}")

    if lolos_f1 and lolos_invalid:
        cfg_final = dict(cfg)
        cfg_final["evaluasi"] = {
            **hasil,
            "iterasi": iterasi,
            "nama_iterasi": NAMA_ITERASI[iterasi],
            "dataset": "Heart Failure Clinical Records (bagian test)",
            "tanggal": datetime.now().strftime("%Y-%m-%d"),
        }
        with open(PROJECT_DIR / "config.json", "w", encoding="utf-8") as file:
            json.dump(cfg_final, file, indent=2, ensure_ascii=False)
        print("LOLOS evaluasi. config.json siap dipakai app.py.")
    else:
        print("BELUM LOLOS. config.json tidak dibuat. Perbaiki prompt di data validasi.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--iterasi", type=int, choices=[1, 2, 3])
    mode.add_argument("--test", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY belum ada. Buat file .env berisi GROQ_API_KEY=gsk_...")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    train, val, pool, test = muat_data()
    print(f"Train: {len(train)} (validasi {len(val)}, pool {len(pool)}) | Test: {len(test)}")

    try:
        if args.iterasi:
            cfg = buat_config(args.iterasi, pool)
            print(f"\nIterasi {args.iterasi}: {NAMA_ITERASI[args.iterasi]} | Model: {cfg['model']}")
            hasil, prediksi = evaluasi(client, cfg, val)
            with open(PROJECT_DIR / f"hasil_validasi_v{args.iterasi}.json", "w",
                      encoding="utf-8") as file:
                json.dump({
                    "iterasi": args.iterasi, "config": cfg, "hasil": hasil,
                    "tanggal": datetime.now().astimezone().isoformat(),
                }, file, indent=2, ensure_ascii=False)
            val.assign(prediksi=prediksi).to_csv(
                PROJECT_DIR / f"prediksi_validasi_v{args.iterasi}.csv", index=False,
            )
            print(f"Tersimpan di hasil_validasi_v{args.iterasi}.json")
            return

        # --test: pilih iterasi dengan F1 validasi tertinggi
        kandidat = []
        for versi in NAMA_ITERASI:
            path = PROJECT_DIR / f"hasil_validasi_v{versi}.json"
            if path.exists():
                with open(path, encoding="utf-8") as file:
                    kandidat.append(json.load(file))
        if not kandidat:
            raise SystemExit("Belum ada hasil validasi. Jalankan --iterasi 1, 2, 3 dulu.")
        print("\nSkor validasi:")
        print(pd.DataFrame({
            f"iterasi_{k['iterasi']}": k["hasil"] for k in kandidat
        }).to_string())
        terbaik = max(kandidat, key=lambda k: k["hasil"]["f1_macro"])
        print(f"\nTerpilih: iterasi {terbaik['iterasi']} ({NAMA_ITERASI[terbaik['iterasi']]})")
        print(f"Evaluasi akhir pada {len(test)} data test...", flush=True)

        hasil_test, prediksi = evaluasi(client, terbaik["config"], test)
        test.assign(prediksi=prediksi).to_csv(
            PROJECT_DIR / "hasil_evaluasi_test.csv", index=False,
        )
        with open(PROJECT_DIR / "hasil_evaluasi_test.json", "w", encoding="utf-8") as file:
            json.dump({
                "iterasi_terpilih": terbaik["iterasi"], "config": terbaik["config"],
                "hasil_validasi": terbaik["hasil"], "hasil_test": hasil_test,
                "tanggal": datetime.now().astimezone().isoformat(),
            }, file, indent=2, ensure_ascii=False)
        finalisasi(hasil_test, terbaik["config"], terbaik["iterasi"])
    finally:
        client.close()


if __name__ == "__main__":
    main()
