import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from classifier import NAMA_FITUR, build_messages

load_dotenv(".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

with open("hasil_evaluasi_test.json", encoding="utf-8") as f:
    cfg = json.load(f)["config"]
hasil = pd.read_csv("hasil_evaluasi_test.csv")
invalid = hasil[hasil["prediksi"] == "invalid"].head(5)

for _, baris in invalid.iterrows():
    data = baris[list(NAMA_FITUR)].to_dict()
    try:
        r = client.chat.completions.create(
            model=cfg["model"], temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"], reasoning_effort=cfg["reasoning_effort"],
            messages=build_messages(cfg, data),
        )
        print(f"id {baris['id']}: jawaban model = {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"id {baris['id']}: ERROR {type(e).__name__}: {str(e)[:200]}")
    time.sleep(4)
client.close()