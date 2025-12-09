import sqlite3
import numpy as np
from openai import OpenAI
from config_api import get_openai_api_key
import os

client = OpenAI(api_key=get_openai_api_key())

DB_PATH = "courses.db"
EMB_PATH = "data/embeddings.npy"
ID_PATH = "data/course_ids.npy"

def fetch_courses():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, description FROM courses ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows

def main():
    courses = fetch_courses()
    print(f"Loaded {len(courses)} courses.")

    texts = [f"{title} {desc}" for (_, title, desc) in courses]

    print("Generating embeddings with OpenAI...")

    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    vectors = np.array([e.embedding for e in resp.data], dtype="float32")
    ids = np.array([cid for (cid, _, _) in courses], dtype="int32")

    os.makedirs("data", exist_ok=True)
    np.save(EMB_PATH, vectors)
    np.save(ID_PATH, ids)

    print(f"Saved embeddings to {EMB_PATH}")
    print(f"Saved course IDs to {ID_PATH}")

if __name__ == "__main__":
    main()
