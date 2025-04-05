import openai
import os

openai.api_base = "https://api.groq.com/openai/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")  # ✅ Must be this

model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

def call_groq_llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Groq API error: {e}]"
