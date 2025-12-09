import sqlite3

DB_PATH = "courses.db"

def get_sql_candidates(keywords):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    conditions = " OR ".join(["title LIKE ? OR description LIKE ?" for _ in keywords])
    params = [f"%{kw}%" for kw in keywords for _ in (0,1)]

    query = f"""
        SELECT id, title, description
        FROM courses
        WHERE {conditions}
    """

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows
