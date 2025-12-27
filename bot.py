import json
import math
import sqlite3
from google import genai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# HARD-CODE KEY (same as ingest.py)
client = genai.Client(
    api_key="YOUR_GEMINI_API_KEY"
)

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
DB_PATH = "rag.db"
EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b)


def embed_query(text: str):
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return res.embeddings[0].values


def retrieve_chunks(question, k=3):
    q_emb = embed_query(question)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT chunk, embedding FROM docs")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for chunk, emb_json in rows:
        emb = json.loads(emb_json)
        score = cosine_similarity(q_emb, emb)
        scored.append((score, chunk))

    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:k]]


def ask_gemini(context, question):
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
    res = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return res.text


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ask <question>")
        return

    question = " ".join(context.args)

    chunks = retrieve_chunks(question)
    context_text = "\n\n".join(chunks)

    answer = ask_gemini(context_text, question)
    await update.message.reply_text(answer)


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("ask", ask))
    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
