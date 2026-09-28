import os
import numpy as np

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Chemin du modèle fine-tuné
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modele_recrutement_finetune"
)


# ============================================================
# Chargement du modèle
# ============================================================

@lru_cache(maxsize=1)
def load_model():

    print("Chargement du modèle fine-tuné...")

    model = SentenceTransformer(
        MODEL_PATH
    )

    print("✅ Modèle fine-tuné chargé.")

    return model


# ============================================================
# Matching TF-IDF + SBERT fine-tuné
# ============================================================

def calculate_matching_score(
    job_description,
    cv_texts
):

    model = load_model()

    # --------------------------------------------------------
    # Sécurité
    # --------------------------------------------------------

    if not cv_texts:

        return [], [], []

    cv_texts = [
        str(text)
        if text is not None
        else ""
        for text in cv_texts
    ]

    job_description = str(
        job_description
        if job_description is not None
        else ""
    )

    # ========================================================
    # 1. TF-IDF
    # ========================================================

    documents = [
        job_description
    ] + cv_texts

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    job_vector = tfidf_matrix[0]

    cv_vectors = tfidf_matrix[1:]

    tfidf_scores = cosine_similarity(
        job_vector,
        cv_vectors
    )[0]

    # Conversion en pourcentage
    tfidf_scores = (
        np.clip(tfidf_scores, 0, 1)
        * 100
    )

    # ========================================================
    # 2. SBERT fine-tuné
    # ========================================================

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    job_embedding = embeddings[0]

    cv_embeddings = embeddings[1:]

    sbert_scores = np.dot(
        cv_embeddings,
        job_embedding
    )

    # Les scores cosinus peuvent théoriquement être négatifs
    sbert_scores = np.clip(
        sbert_scores,
        0,
        1
    ) * 100

    # ========================================================
    # 3. Score final
    # ========================================================

    final_scores = (
        0.30 * tfidf_scores
        + 0.70 * sbert_scores
    )

    return (
        tfidf_scores.tolist(),
        sbert_scores.tolist(),
        final_scores.tolist()
    )