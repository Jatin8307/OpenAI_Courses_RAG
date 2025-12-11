import streamlit as st
from search_courses_stage1 import get_sql_candidates
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from semantic_ranker_LLM import rank_with_llm 
from reranker_local import heuristic_rerank, rerank_llm
from config_api import FAISS_DIR, LOCAL_EMBEDDING_MODEL
import numpy as np

st.set_page_config(page_title="Smart Course Recommender", layout="wide")

@st.cache_resource
def load_faiss():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}  # Top 20 candidates for reranker
    )
    return retriever

retriever = load_faiss()

st.title("Smart Course Recommender")
st.write("Hybrid SQL + RAG + LLM Re-Ranker")

query_input = st.text_input("Enter course topic:", placeholder="e.g. web development, react, aws, python...")

if st.button("Search"):

    if not query_input.strip():
        st.warning("Please enter something to search.")
        st.stop()

    query = query_input.strip()
    keywords = [k.lower() for k in query.split()]

    # -------- Stage 1: SQL Exact Match --------
    sql_results = get_sql_candidates(keywords)

    if sql_results:
        st.subheader("SQL Exact Matches:")
        for c in sql_results[:10]:
            st.write(f"### {c['title']}")
            st.write(c["description"])
            st.divider()
    else:
        st.warning("No SQL matches. Switching to AI-based search...")

    # -------- Stage 2: RAG Retrieval --------
    st.write("Retrieving semantically similar courses...")
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs:
        st.error("No relevant semantic matches found.")
        st.stop()

    # Convert LangChain docs → simple dicts
    rag_candidates = [
        {
            "id": doc.metadata["id"],
            "title": doc.metadata["title"],
            "description": doc.metadata["description"],
        }
        for doc in retrieved_docs
    ]

    # -------- Stage 3: LLM Re-Ranking --------
    st.write("Re-ranking using AI (LLM)...")
    local_rank = heuristic_rerank(query, rag_candidates)
    if len(local_rank) < 5:   # too weak → try LLM
        final_results = rerank_llm(query, rag_candidates)
    else:
        final_results = local_rank

    if not final_results:
        st.error("AI could not rank results.")
        st.stop()

    # -------- Display Final Results --------
    st.subheader("Top Recommended Courses:")

    for i, course in enumerate(final_results, start=1):
        st.markdown(f"### {i}. {course['title']}")
        st.write(course.get("description", ""))
        st.divider()
