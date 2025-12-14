import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)



def openai_rerank(candidates, query, max_output=10):
    """Rerank semantic candidates using OpenAI GPT-4o model."""

    if not OPENAI_API_KEY:
        print("OpenAI API key missing. Using heuristic rerank instead.")
        return None

    # Prepare text representation for LLM
    candidate_strings = []
    for i, c in enumerate(candidates, start=1):
        candidate_strings.append(
            f"[{i}] Title: {c['title']}\nDescription: {c['description']}"
        )

    joined = "\n\n".join(candidate_strings)

    prompt = f"""
You are a ranking engine. Rank the following courses for query: "{query}".
Return ONLY the numbers of the best {max_output} matching courses in order.

Courses:
{joined}

Your output format should be EXACTLY like:
1, 5, 3, 2
(no extra text)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        chosen_ids = [
            int(x.strip()) for x in content.replace("\n", "").split(",") if x.strip().isdigit()
        ]
    except Exception as e:
        print("LLM rerank failed:", e)
        return None

    # Map back to original objects
    final = []
    for idx in chosen_ids:
        if 1 <= idx <= len(candidates):
            final.append(candidates[idx - 1])
    # print("openi ai me enter kiya tha")
    return final[:max_output]


def heuristic_rerank(candidates, max_output=10):
    """Simple fallback when LLM fails."""
    # print("Using heuristic rerank.")
    sorted_items = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_items[:max_output]
