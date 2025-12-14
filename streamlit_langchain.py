import streamlit as st
import sqlite3
from dotenv import load_dotenv
load_dotenv()

from pinecone_retriever_adaptor import get_retriever
from llm_reranker import openai_rerank, heuristic_rerank

st.set_page_config(page_title="Smart Course Recommender", layout="wide")
st.title("Smart Course Recommender — LangChain RAG (Pinecone)")

query = st.text_input("Enter topic / course / skill (e.g., web development, singing, aws):")

if st.button("Search"):
    q = (query or "").strip()
    if not q:
        st.warning("Please enter a query.")
        st.stop()

    # Stage 1 - SQL exact/LIKE (optional hybrid)
    try:
        conn = sqlite3.connect("courses.db")
        cur = conn.cursor()
        keywords = [f"%{k.strip().lower()}%" for k in q.split() if k.strip()]
        rows = []
        if keywords:
            where = " OR ".join(["(LOWER(title) LIKE ? OR LOWER(description) LIKE ?)"] * len(keywords))
            params = []
            for k in keywords:
                params += [k, k]
            sql = f"SELECT id, title, description, COALESCE(category,''), COALESCE(level,'') FROM courses WHERE {where} LIMIT 40"
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.close()
    except Exception:
        rows = []

    if rows:
        st.success(f"Found {len(rows)} SQL matches")
        for i, (cid, title, desc, cat, level) in enumerate(rows, start=1):
            st.markdown(f"### {i}. {title}")
            st.write(desc)
            st.caption(f"Category: {cat}  •  Level: {level}")
            st.divider()

    else:
        st.info("No direct SQL matches — using semantic RAG fallback.")
        with st.spinner("Searching via embeddings + Pinecone..."):
            retr = get_retriever(k=60)
            docs = retr.get_relevant_documents(q)

        if not docs:
            st.error("No semantic matches found in the database.")
            st.stop()

        # convert docs -> candidates (dicts)
        candidates = []
        for d in docs:
            meta = d.metadata or {}
            candidates.append({
                "id": meta.get("id"),
                "title": meta.get("title"),
                "description": d.page_content,
                "category": meta.get("category"),
                "level": meta.get("level"),
                "score": meta.get("score", 0.0)
            })

        st.write(f"Retrieved {len(candidates)} semantic candidates from Pinecone.")

        with st.spinner("Reranking candidates (LLM)..."):
            final = openai_rerank(candidates, q, max_output=10)

        if not final:
            final = heuristic_rerank(candidates, max_output=10)

        st.success("Top AI-recommended courses:")
        for i, c in enumerate(final, start=1):
            st.markdown(f"### {i}. {c['title']}  (score: {c.get('score'):.4f})")
            st.write(c.get("description",""))
            st.caption(f"Category: {c.get('category')}  •  Level: {c.get('level')}")
            st.divider()
