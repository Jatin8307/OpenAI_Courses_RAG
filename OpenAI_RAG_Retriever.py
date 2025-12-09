import sqlite3
import numpy as np
from openai import OpenAI
from config_api import get_openai_api_key

client = OpenAI(api_key=get_openai_api_key())

DB_PATH = "courses.db"
EMB_PATH = "data/embeddings.npy"
ID_PATH = "data/course_ids.npy"

# Load vectors & IDs
VECTORS = np.load(EMB_PATH)
COURSE_IDS = np.load(ID_PATH)

def load_course_map():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, description FROM courses")
    rows = cur.fetchall()
    conn.close()
    return {cid: (cid, title, desc) for (cid, title, desc) in rows}

COURSE_MAP = load_course_map()

def embed_query(text):
    e = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    ).data[0].embedding
    return np.array(e, dtype="float32")

def retrieve_top_k(query, k=40):
    q_vec = embed_query(query)

    q_norm = q_vec / np.linalg.norm(q_vec)
    v_norm = VECTORS / np.linalg.norm(VECTORS, axis=1, keepdims=True)

    sims = np.dot(v_norm, q_norm)
    idx = np.argsort(-sims)[:k]

    results = [(COURSE_IDS[i], *COURSE_MAP[COURSE_IDS[i]][1:]) for i in idx]
    scores = [float(sims[i]) for i in idx]
    return results, scores
