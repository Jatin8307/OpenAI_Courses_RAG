from search_courses_stage1 import get_sql_candidates
from retrievel_local import hybrid_retrieve
from reranker_local import rerank

def run_search_pipeline(query: str, use_sql_first=True, metadata_filter=None):
    keywords = [t.strip().lower() for t in query.split() if t.strip()]
    sql_results = get_sql_candidates(keywords) if use_sql_first else []
    if sql_results:
        # SQL returned matches: convert to dict format for display
        return [{"id": r[0], "title": r[1], "description": r[2]} for r in sql_results], "sql"
    # else run hybrid retriever (vector fallback)
    candidates = hybrid_retrieve(query, keywords, k=80, metadata_filter=metadata_filter, sql_first=False)
    if not candidates:
        return [], "none"
    final = rerank(query, candidates, max_results=10)
    return final, "rag"
