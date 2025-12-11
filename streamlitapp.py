# streamlit_local_rag.py
import streamlit as st
from pipeline_local_rag import run_search_pipeline
from config_api import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
st.set_page_config(page_title="Local RAG — FAISS + SQL", layout="wide")
st.title("Smart Course Recommender powered by RAG")

query = st.text_input("Enter topic, skill, or course name:", placeholder="e.g. web development, aws, singing")
use_sql_first = st.checkbox("First SQL matches", value=True)
use_metadata = st.checkbox("Apply metadata filter", value=False)

metadata_filter = None
if use_metadata:
    cat = st.text_input("Category (exact match)")
    lvl = st.selectbox("Level", ["", "Beginner","Intermediate","Advanced","Professional","Crash Course"])
    mf = {}
    if cat: mf["category"] = cat
    if lvl: mf["level"] = lvl
    if mf:
        metadata_filter = mf

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query")
        st.stop()

    with st.spinner("Running pipeline..."):
        results, source = run_search_pipeline(query, use_sql_first=use_sql_first, metadata_filter=metadata_filter)

    if source == "sql":
        st.success(f"SQL found {len(results)} results (exact matches).")
        for i, r in enumerate(results[:20], start=1):
            st.markdown(f"### {i}. {r['title']}")
            st.write(r.get("description",""))
            st.divider()
    elif source == "rag":
        st.success(f"RAG returned {len(results)} recommendations.")
        for i, r in enumerate(results, start=1):
            st.markdown(f"### {i}. {r.get('title')}")
            st.write(r.get("description",""))
            cat = r.get("category") or "Unknown"
            lvl = r.get("level") or ""
            st.caption(f"Category: {cat} — Level: {lvl}")
            st.divider()
    else:
        st.error("No results found in DB or vector store.")
