import os, sqlite3
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "courses.db")
MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "courses-index")
DIM = 384  # all-MiniLM-L6-v2 dim

print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# Pinecone v5 client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not present
existing = pc.list_indexes().names()
if INDEX_NAME not in existing:
    print("Creating index:", INDEX_NAME)
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Read DB
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT id, title, description, COALESCE(category,''), COALESCE(level,'') FROM courses")
rows = cur.fetchall()
conn.close()

print(f"Found {len(rows)} rows. Upserting into Pinecone in batches...")
batch = []
BATCH_SIZE = 100

for cid, title, desc, cat, level in tqdm(rows):
    text = f"{title}\n{desc}"
    vec = model.encode(text).astype("float32").tolist()
    metadata = {"id": str(cid), "title": title, "description": desc, "category": cat, "level": level}
    batch.append({"id": str(cid), "values": vec, "metadata": metadata})

    if len(batch) >= BATCH_SIZE:
        index.upsert(vectors=batch)
        batch = []

if batch:
    index.upsert(vectors=batch)

print("Ingestion complete.")
