import streamlit as st
from search_courses_stage1 import get_sql_candidates
from OpenAI_RAG_Retriever import retrieve_top_k
from semantic_ranker_LLM import rank_with_llm

st.set_page_config(page_title="Course Search by RAG", layout="wide")
st.title("Smart Course Search (SQL + OpenAI RAG)")

query = st.text_input("Search for any skill, topic, domain, or course:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    keywords = [k.lower() for k in query.split()]
    sql_results = get_sql_candidates(keywords)

    if sql_results:
        st.success(f"Found {len(sql_results)} SQL matches.")

        for cid, title, desc in sql_results[:20]:
            st.markdown(f"### {title}")
            st.write(desc)
            st.divider()
    else:
        st.warning("No SQL matches. Switching to AI semantic search...")

        rag_results, scores = retrieve_top_k(query, k=40)
        final = rank_with_llm(query, rag_results)

        st.success("AI Recommended Courses:")
        for cid, title, desc in final:
            st.markdown(f"### {title}")
            st.write(desc)
            st.divider()
