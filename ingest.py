import os
import json
import sqlite3
from google import genai

# HARD-CODE KEY (for now)
client = genai.Client(
    api_key="YOUR_GEMINI_API_KEY"
)

DB_PATH = "rag.db"
DOCS_PATH = "docs"
EMBED_MODEL = "text-embedding-004"


def chunk_text(text, size=400):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


def embed(text: str):
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return response.embeddings[0].values


def ingest():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk TEXT,
            embedding TEXT
        )
    """)

    for file in os.listdir(DOCS_PATH):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            emb = embed(chunk)
            cur.execute(
                "INSERT INTO docs (chunk, embedding) VALUES (?, ?)",
                (chunk, json.dumps(emb))
            )

    conn.commit()
    conn.close()
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
