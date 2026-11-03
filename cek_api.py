import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
try:
    r = client.chat.completions.create(
        model="qwen/qwen3.8-27b", max_tokens=5, reasoning_effort="none",
        messages=[{"role": "user", "content": "Jawab: OK"}],
    )
    print("API normal. Jawaban:", r.choices[0].message.content)
except Exception as e:
    print("ERROR:", type(e).__name__)
    print(e)