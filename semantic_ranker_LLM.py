from openai import OpenAI
from config_api import get_openai_api_key
import json

client = OpenAI(api_key=get_openai_api_key())

def rank_with_llm(query, candidates, max_results=10):

    formatted = "\n".join([
        f"{cid}. {title} - {desc[:120]}..."
        for (cid, title, desc) in candidates
    ])

    prompt = f"""
User query: "{query}"

Below is a list of course titles and descriptions.
Your tasks:
1. Remove irrelevant courses.
2. Remove duplicates.
3. Ensure topic diversity when possible.
4. Rank by relevance and usefulness.
5. Return ONLY ranked course IDs (max {max_results}).

Courses:
{formatted}

Return valid JSON only:
{{"ranked_ids": [ID1, ID2, ...]}}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    try:
        ranked_ids = json.loads(resp.choices[0].message.content)["ranked_ids"]
    except:
        return candidates[:max_results]

    id_map = {cid: (cid, title, desc) for (cid, title, desc) in candidates}
    final = [id_map[cid] for cid in ranked_ids if cid in id_map]

    return final[:max_results]
