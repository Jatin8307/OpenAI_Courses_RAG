# reranker_local.py
from typing import List, Dict, Any
from config_api import get_openai_api_key
import re
OPENAI_API_KEY = get_openai_api_key()
# optional: use OpenAI if key present (you said local; this is optional)
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        _client = None
except Exception:
    _client = None

def heuristic_rerank(query: str, candidates: List[Dict[str, Any]], max_results: int = 10):
    seen_titles = set()
    final = []
    for c in candidates:
        title_norm = re.sub(r"\W+", " ", (c.get("title") or "").lower()).strip()
        if title_norm in seen_titles:
            continue
        seen_titles.add(title_norm)
        final.append(c)
        if len(final) >= max_results:
            break
    return final[:max_results]

def openai_rerank(query: str, candidates: List[Dict[str, Any]], max_results: int = 10):
    if _client is None:
        return heuristic_rerank(query, candidates, max_results)
    # Build prompt
    items = []
    for c in candidates[:100]:
        items.append(f"{c['id']}. {c.get('title','')[:120]} -- {(c.get('description') or '')[:200].replace(chr(10),' ')}")
    prompt = f"""
You are an assistant that ranks candidate course items for a user query.
Query: {query}

Candidates:
{chr(10).join(items)}

Remove irrelevant items, deduplicate near-duplicates, ensure topic diversity and return JSON:
{{"ranked_ids": [id1,id2,...]}} (max {max_results} ids)
"""
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=200
    )
    raw = resp.choices[0].message.content
    import json
    try:
        ranked = json.loads(raw)["ranked_ids"]
    except Exception:
        ranked = [int(x) for x in re.findall(r"\d+", raw)]
    idmap = {c["id"]: c for c in candidates}
    out = [idmap[r] for r in ranked if r in idmap]
    if not out:
        return heuristic_rerank(query, candidates, max_results)
    return out[:max_results]

def rerank(query: str, candidates: List[Dict[str, Any]], max_results:int=10):
    # prefer heuristic (fast & offline)
    try:
        return heuristic_rerank(query, candidates, max_results)
    except Exception:
        return candidates[:max_results]
