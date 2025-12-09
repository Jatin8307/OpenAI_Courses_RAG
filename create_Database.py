"""
create_courses_db.py

Creates courses.db and populates it with 2000 diverse, non-duplicate courses.
Schema:
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  category TEXT,
  level TEXT

"""

import sqlite3
import random
import textwrap

DB_PATH = "courses.db"
NUM_COURSES = 2000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Categories and level pools to ensure broad coverage
CATEGORIES = [
    "Programming", "Web Development", "Data Science", "Machine Learning", "NLP",
    "Cloud Computing", "AWS", "Azure", "GCP", "DevOps", "Cybersecurity",
    "Databases", "Business Intelligence", "Product Management", "UX/UI",
    "Mobile Development", "Game Development", "Blockchain", "Finance",
    "Marketing", "Social Media", "Photography", "Video Production", "Music",
    "Languages", "Health & Fitness", "Personal Development", "Math", "Physics",
    "Chemistry", "Biology", "Education", "Cooking", "Legal", "Ethics",
    "Design", "3D Modeling", "AR/VR", "Robotics", "Electronics", "IoT",
    "Statistics", "Excel", "Tableau", "Power BI", "Quantum Computing", "Astronomy",
    "Environmental Science", "Sustainability", "Public Speaking", "Writing",
    "Leadership"
]

LEVELS = ["Beginner", "Intermediate", "Advanced", "Professional", "Crash Course"]

# Adjective / focus fragments to make titles diverse and realistic
TITLE_PATTERNS = [
    "{skill}: A Practical Guide",
    "Mastering {skill} - From Fundamentals to Production",
    "Hands-on {skill} Projects",
    "{skill} for {audience}",
    "The Complete {skill} Bootcamp",
    "Introduction to {skill}",
    "{skill} - Real-World Use Cases",
    "Advanced {skill} Techniques",
    "{skill} in Practice: Case Studies",
    "Rapid {skill} Workshop",
    "Applied {skill} - Industry Examples",
    "{skill} for Professionals",
    "Comprehensive {skill} Toolkit",
    "{skill}: From Zero to Hero",
    "{skill} Optimization and Best Practices",
]

# Short templates to craft varied, realistic descriptions (different tones)
DESCRIPTION_TEMPLATES = [
    "This course offers a hands-on introduction to {skill}. You will build practical projects, learn common pitfalls, and leave with a portfolio-ready project.",
    "Designed for {audience}, this program dives deep into {skill} techniques used in industry-grade systems. Includes guided exercises and a capstone.",
    "{skill} fundamentals explained with real-world examples. Expect focused lessons, code-along notebooks, and practical assessments.",
    "Explore advanced {skill} strategies and optimizations. Ideal for practitioners who want to scale {skill} solutions in production.",
    "A project-driven course that covers core concepts, troubleshooting, and performance tuning for {skill}. Suitable for learners who prefer applied learning.",
    "Learn {skill} from the ground up: conceptual foundations, hands-on labs, and one end-to-end case study demonstrating best practices.",
    "Short, intensive workshop on {skill} that concentrates on high-impact patterns and immediate productivity gains.",
    "A deep dive into {skill} covering both theoretical foundations and practical deployment scenarios with modern toolchains.",
    "An industry-oriented curriculum centered on {skill}, complete with code reviews, deployment checklists, and real engineering challenges.",
    "This course emphasizes building production-ready {skill} solutions, including testing, monitoring, and cost optimization.",
]

# Skills pool to combine with categories
SKILLS = [
    "Python", "JavaScript", "React", "Node.js", "Django", "Flask",
    "Rust systems programming", "Golang microservices", "Kotlin Android", "Swift iOS",
    "SQL Performance Tuning", "NoSQL Modeling", "Postgres Internals", "MongoDB Schemas",
    "Data Pipelines with Airflow", "ETL Engineering", "Pandas Advanced", "Feature Engineering",
    "PyTorch Fundamentals", "TensorFlow for Production", "Transformer Models", "Prompt Engineering",
    "Natural Language Understanding", "Speech Recognition", "Computer Vision",
    "AWS EC2 & Infrastructure", "AWS Lambda Architectures", "Azure DevOps", "GCP Dataflow",
    "Kubernetes in Production", "Docker Best Practices", "CI/CD with GitHub Actions",
    "Site Reliability Engineering", "Infrastructure as Code (Terraform)",
    "Penetration Testing Basics", "Network Security", "Identity and Access Management",
    "Tableau Dashboards", "Power BI Reporting", "Looker Studio",
    "Product Roadmapping", "A/B Testing", "Growth Hacking", "SEO Strategy",
    "UX Research", "Design Systems", "Figma UI Patterns", "Accessibility Best Practices",
    "React Native", "Flutter Cross-Platform", "Unity Game Mechanics",
    "Solidity Smart Contracts", "Blockchain Architecture", "NFT Design",
    "Quantitative Finance Modeling", "Excel for Analysts", "Financial Statement Analysis",
    "Photography: Lighting & Color", "Video Editing with Premiere", "Sound Design",
    "Guitar Fundamentals", "Vocal Technique", "Spanish for Beginners", "Business English",
    "Public Speaking Confidence", "Mindfulness for Productivity", "Yoga Foundations",
    "3D Modeling with Blender", "AR/VR Interaction Design", "Robotics with ROS 2",
    "Embedded Systems with Arduino", "IoT Sensor Networks", "Renewable Energy Basics",
    "Quantum Programming with Qiskit", "Astronomy Observation Techniques",
    "Sustainable Business Models", "Ethical AI", "Healthcare Analytics",
    "Legal Contracts for Startups", "Technical Writing for Engineers"
]

# Helper functions to construct unique, varied titles/descriptions
def pick_category_and_skill(i):
    # bias distribution so many categories get multiple examples but still broad coverage
    category = random.choice(CATEGORIES)
    skill = random.choice(SKILLS)
    # make some titles use category-specific phrasing (e.g., AWS courses)
    if category in ["AWS", "Azure", "GCP", "Cloud Computing"] and random.random() < 0.6:
        skill = random.choice([
            "AWS Cloud Architecture", "Serverless on AWS", "GCP Data Engineering",
            "Azure Machine Learning Pipeline"
        ])
    if category in ["DevOps", "Cloud Computing"] and random.random() < 0.4:
        skill = random.choice(["Kubernetes in Production", "Terraform Infrastructure"])
    return category, skill

def unique_title(i, skill):
    pattern = random.choice(TITLE_PATTERNS)
    # incorporate index lightly to ensure no collisions but keep naturalness
    modifier = "" if random.random() < 0.85 else f" — Practicum {i%50 + 1}"
    # sometimes use a sub-audience
    audience = random.choice(["for Engineers", "for Managers", "for Data Analysts", "for Creators", "for Beginners", "for Entrepreneurs"])
    title = pattern.format(skill=skill, audience=audience) + modifier
    # avoid super long titles
    return title[:140]

def unique_description(skill, category, level, i):
    template = random.choice(DESCRIPTION_TEMPLATES)

    audience_phrase = {
        "Beginner": "No prior experience required.",
        "Intermediate": "Some prior experience recommended.",
        "Advanced": "Advanced knowledge expected.",
        "Professional": "Designed for working professionals.",
        "Crash Course": "Fast-paced, outcome oriented."
    }[level]

    modules = random.sample([
        "Foundations and concepts", "Hands-on labs", "Project 1: Build an app",
        "Scalability & performance", "Testing & CI", "Monitoring & ops",
        "Capstone project", "Case studies from industry", "Security best-practices",
        "Data modeling and ETL", "UX critiques", "Deployment & costs", "Optimization"
    ], k=3)

    extra = " Modules include: " + "; ".join(modules) + "."

    desc = template.format(
        skill=skill,
        audience="professionals" if level == "Professional" else "students"
    )

    full = (
        f"{desc} {audience_phrase} {extra} "
        f"Typical course length: {8 + (i % 12)} weeks."
    )

    return textwrap.shorten(full.replace("\n", " "), width=600, placeholder="...")


# Build courses and insert into SQLite
def build_and_insert(db_path=DB_PATH, n=NUM_COURSES):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # recreate table
    cur.execute("DROP TABLE IF EXISTS courses")
    cur.execute("""
    CREATE TABLE courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT,
        level TEXT
    )
    """)
    conn.commit()

    # levels cycling gives variety
    level_cycle = ["Beginner", "Intermediate", "Advanced", "Professional", "Crash Course"]

    inserted_titles = set()
    rows = []
    i = 0
    attempts = 0
    while i < n and attempts < n * 4:
        attempts += 1
        category, skill = pick_category_and_skill(i)
        level = level_cycle[i % len(level_cycle)]
        title = unique_title(i, skill)
        # ensure title uniqueness (no near duplicates)
        if title.lower() in inserted_titles:
            # small chance to mutate title if collision
            title = title + f" ({random.choice(['Workshop','Lab','Series','Intensive'])})"
            if title.lower() in inserted_titles:
                continue
        desc = unique_description(skill, category, level,i)
        # further ensure content diversity: slightly vary descriptions for same skill
        desc = desc.replace(skill, skill + ("" if random.random() < 0.7 else " — practical edition"))
        # record
        inserted_titles.add(title.lower())
        rows.append((title, desc, category, level))
        i += 1

    # bulk insert
    cur.executemany("INSERT INTO courses (title, description, category, level) VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    print(f"Inserted {len(rows)} courses into {db_path}")

if __name__ == "__main__":
    build_and_insert()
