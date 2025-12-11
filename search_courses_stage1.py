import sqlite3
from config_api import DB_PATH

def get_sql_candidates(keywords):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Basic LIKE search
    conditions = []
    params = []
    for kw in keywords:
        conditions.append("(title LIKE ? OR description LIKE ?)")
        params.append(f"%{kw}%")
        params.append(f"%{kw}%")

    if not conditions:
        return []

    query = f"""
        SELECT id, title, description
        FROM courses
        WHERE {" OR ".join(conditions)}
        LIMIT 20
    """

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    # ---- FIX: Convert tuple → dict ----
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "title": r[1],
            "description": r[2]
        })

    return results
