# search_courses_stage1.py
import sqlite3
from typing import List, Tuple

DB_PATH = "courses.db"

def get_sql_candidates(keywords: List[str], limit: int = 20) -> List[Tuple[int, str, str]]:
    """
    Simple SQL LIKE search over title and description.
    Returns list of tuples: (id, title, description)
    """
    if not keywords:
        return []

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    conds = " OR ".join(["(title LIKE ? OR description LIKE ?)" for _ in keywords])
    params = []
    for k in keywords:
        kw = f"%{k}%"
        params.extend([kw, kw])

    query = f"""
        SELECT id, title, description
        FROM courses
        WHERE {conds}
        LIMIT ?
    """
    cur.execute(query, params + [limit])
    rows = cur.fetchall()
    conn.close()
    return rows
