from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_matching_score(job_offer, cv_texts):
    """
    Calcule le score de correspondance entre une offre d'emploi
    et plusieurs CV à l'aide de TF-IDF et de la similarité cosinus.

    Paramètres
    ----------
    job_offer : str
        Texte de l'offre d'emploi.

    cv_texts : list
        Liste contenant les textes des CV.

    Retour
    ------
    list
        Liste des scores de correspondance sur 100.
    """

    # Vérification de l'offre
    if not job_offer or not job_offer.strip():
        raise ValueError("L'offre d'emploi est vide.")

    # Vérification des CV
    if not cv_texts:
        raise ValueError("Aucun CV n'a été fourni.")

    # Remplacer les CV vides par un texte neutre
    cleaned_cv_texts = []

    for cv in cv_texts:
        if cv and cv.strip():
            cleaned_cv_texts.append(cv)
        else:
            cleaned_cv_texts.append("aucune information disponible")

    # Ensemble des documents à comparer
    documents = [job_offer] + cleaned_cv_texts

    # Création du modèle TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )

    # Transformation des textes en vecteurs numériques
    vectors = vectorizer.fit_transform(documents)

    # Premier vecteur = offre d'emploi
    job_vector = vectors[0:1]

    # Les autres vecteurs = CV
    cv_vectors = vectors[1:]

    # Calcul de la similarité entre l'offre et chaque CV
    similarities = cosine_similarity(
        job_vector,
        cv_vectors
    )[0]

    # Conversion en pourcentage
    scores = similarities * 100

    # Arrondi à 2 chiffres
    scores = [round(float(score), 2) for score in scores]

    return scores