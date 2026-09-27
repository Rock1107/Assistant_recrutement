import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Modèle utilisé pour transformer les textes en vecteurs
model = SentenceTransformer("all-MiniLM-L6-v2")


def prepare_documents(offres, candidatures):
    """
    Transforme les offres d'emploi et les CV
    en documents utilisables par le système RAG.
    """

    documents = []
    metadata = []

    # -------------------------
    # Documents des offres
    # -------------------------
    for _, offre in offres.iterrows():

        document = f"""
        Offre d'emploi : {offre['poste']}

        Description :
        {offre['description']}

        Compétences requises :
        {offre['competences']}

        Niveau d'études :
        {offre['niveau_etudes']}

        Expérience requise :
        {offre['experience_requise']}
        """

        documents.append(document)

        metadata.append({
            "type": "offre",
            "id": offre["id_offre"],
            "nom": offre["poste"]
        })

    # -------------------------
    # Documents des candidats
    # -------------------------
    for _, candidat in candidatures.iterrows():

        document = f"""
        Candidat : {candidat['nom']}

        Poste recherché :
        {candidat['poste_cible']}

        Diplôme :
        {candidat['diplome']}

        Competences :
        {candidat['competences']}

        Expérience :
        {candidat['experience']}

        Certifications :
        {candidat['certifications']}

        CV :
        {candidat['cv_text']}
        """

        documents.append(document)

        metadata.append({
            "type": "candidat",
            "id": candidat["candidate_id"],
            "nom": candidat["nom"]
        })

    return documents, metadata


def create_rag_index(documents):
    """
    Transforme les documents en embeddings
    puis crée l'index FAISS.
    """

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    # Recherche par similarité cosinus
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, embeddings


def search_rag(question, documents, metadata, index, top_k=3):
    """
    Recherche les documents les plus pertinents
    pour une question donnée.
    """

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, index_document in zip(
        scores[0],
        indices[0]
    ):

        results.append({
            "score": float(score),
            "document": documents[index_document],
            "metadata": metadata[index_document]
        })

    return results