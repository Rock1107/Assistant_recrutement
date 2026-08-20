from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# --------------------------------
# CHARGEMENT DU MODELE SBERT
# --------------------------------
@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# --------------------------------
# CALCUL DU MATCHING
# --------------------------------

def calculate_matching_score(job_text, cv_texts):

    # TF-IDF
    documents = [job_text] + cv_texts

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    tfidf_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    tfidf_scores = [
        score * 100
        for score in tfidf_scores
    ]

    # Sentence-BERT
    embeddings = model.encode(
        documents,
        convert_to_tensor=True
    )

    job_embedding = embeddings[0]
    cv_embeddings = embeddings[1:]

    sbert_scores = cosine_similarity(
        job_embedding.cpu().numpy().reshape(1, -1),
        cv_embeddings.cpu().numpy()
    )[0]

    sbert_scores = [
        score * 100
        for score in sbert_scores
    ]

    # Combinaison TF-IDF + SBERT
    final_scores = []

    for tfidf, sbert in zip(
        tfidf_scores,
        sbert_scores
    ):

        final_score = (
            0.4 * tfidf +
            0.6 * sbert
        )

        final_scores.append(
            round(final_score, 2)
        )

    return (
        tfidf_scores,
        sbert_scores,
        final_scores
    )