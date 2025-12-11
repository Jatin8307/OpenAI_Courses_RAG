import numpy as np
from openai import OpenAI
from config_api import get_openai_api_key

client = OpenAI(api_key=get_openai_api_key())

# -----------------------------
# 1) Local Heuristic Reranker
# -----------------------------
def heuristic_rerank(query, candidates):
    """
    Simple keyword-based + diversity reranker.
    """
    query_words = set(query.lower().split())

    scored = []
    for c in candidates:
        title = c["title"].lower()
        desc = c["description"].lower()

        # keyword match score
        kw_score = sum(1 for w in query_words if w in title or w in desc)

        # length penalty (prefer detailed descriptions)
        len_score = min(len(desc) / 150, 1.0)

        total = kw_score + len_score
        scored.append((total, c))

    # sort descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # return top 10 unique
    return [c for _, c in scored][:10]


# -----------------------------
# 2) OpenAI LLM Re-Ranker
# -----------------------------
def rerank_llm(query, candidates, max_items=10):
    """
    Ask LLM to re-rank only the retrieved set.
    """
    items = [
        f"{c['id']}. {c['title']} — {c['description'][:60]}..."
        for c in candidates
    ]
    course_block = "\n".join(items)

    prompt = f"""
You are a strict course ranking AI.
User query: "{query}"

Below is a list of course IDs with titles and descriptions.
Filter irrelevant courses.
Remove duplicates.
Return ONLY the relevant IDs sorted by best match.

Return JSON:
{{"ranked_ids": [3, 7, 12, ...] }}

Courses:
{course_block}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        import json
        ranked_ids = json.loads(resp.choices[0].message.content)["ranked_ids"]

        id_to_course = {c["id"]: c for c in candidates}
        final = [id_to_course[i] for i in ranked_ids if i in id_to_course]

        return final[:max_items]

    except:
        # fallback to heuristic
        return heuristic_rerank(query, candidates)
