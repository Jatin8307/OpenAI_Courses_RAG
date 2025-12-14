import sqlite3
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "course-index")
MODEL = SentenceTransformer(os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Load DB
conn = sqlite3.connect("courses.db")
cur = conn.cursor()

cur.execute("SELECT id, title, description FROM courses")
rows = cur.fetchall()

vectors = []

for cid, title, desc in tqdm(rows):
    text = f"{title}. {desc}"
    emb = MODEL.encode(text).tolist()

    vectors.append({
        "id": str(cid),
        "values": emb,
        "metadata": {"title": title, "description": desc}
    })

# Upload to Pinecone
index.upsert(vectors)

print("✅ Pinecone ingest completed!")
