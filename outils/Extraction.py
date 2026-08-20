import re


SKILLS = [
    "python",
    "sql",
    "power bi",
    "excel",
    "tableau",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "spark",
    "pyspark",
    "nlp",
    "llm",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "postgresql",
    "mysql",
    "mongodb",
    "java",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "flask",
    "fastapi",
    "streamlit"
]
def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found_skills.append(skill)
    return sorted(set(found_skills))