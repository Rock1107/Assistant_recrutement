import streamlit as st
import pandas as pd

from outils.pdf_parser import extract_pdf_text
from outils.Extraction import extract_skills
from outils.Matching import calculate_matching_score


# --------------------------------
# CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Assistant intelligent de recrutement",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------
# TITRE
# --------------------------------

st.title("🤖 Assistant intelligent de recrutement")

st.write(
    "Analyse automatique des CV et matching avec une offre d'emploi."
)


# --------------------------------
# OFFRE
# --------------------------------

st.header("1️⃣ Offre d'emploi")

job_title = st.text_input(
    "Intitulé du poste",
    placeholder="Exemple : Data Analyst"
)

job_description = st.text_area(
    "Description de l'offre",
    height=200,
    placeholder=(
        "Exemple :\n"
        "Nous recherchons un Data Analyst maîtrisant "
        "Python, SQL, Power BI et Excel."
    )
)


# --------------------------------
# CV
# --------------------------------

st.header("2️⃣ CV des candidats")

uploaded_files = st.file_uploader(
    "Importer les CV",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------
# BOUTON ANALYSE
# --------------------------------

if st.button(
    "🚀 Analyser les candidatures",
    type="primary"
):

    if not job_description:
        st.error(
            "Veuillez saisir la description de l'offre."
        )
        st.stop()

    if not uploaded_files:
        st.error(
            "Veuillez importer au moins un CV."
        )
        st.stop()


    # --------------------------------
    # EXTRACTION DES CV
    # --------------------------------

    candidates = []
    cv_texts = []

    for file in uploaded_files:

        text = extract_pdf_text(file)

        skills = extract_skills(text)

        candidates.append({
            "Candidat": file.name,
            "Compétences": ", ".join(skills),
            "Texte": text
        })

        cv_texts.append(text)


    # --------------------------------
    # MATCHING TF-IDF + SBERT
    # --------------------------------

    tfidf_scores, sbert_scores, final_scores = calculate_matching_score(
        job_description,
        cv_texts
    )


    # --------------------------------
    # AJOUT DES SCORES
    # --------------------------------

    for candidate, tfidf, sbert, final in zip(
        candidates,
        tfidf_scores,
        sbert_scores,
        final_scores
    ):

        candidate["TF-IDF"] = round(
            float(tfidf),
            2
        )

        candidate["SBERT"] = round(
            float(sbert),
            2
        )

        candidate["Score"] = round(
            float(final),
            2
        )


    # --------------------------------
    # CLASSEMENT
    # --------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x["Score"],
        reverse=True
    )


    # --------------------------------
    # RESULTATS
    # --------------------------------

    st.success(
        "Analyse terminée avec succès."
    )

    st.header("3️⃣ Classement des candidats")

    results = pd.DataFrame(candidates)

    display_results = results[
        [
            "Candidat",
            "Compétences",
            "TF-IDF",
            "SBERT",
            "Score"
        ]
    ]

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------
    # MEILLEUR CANDIDAT
    # --------------------------------

    best = candidates[0]

    st.header("🏆 Meilleur candidat")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Candidat",
            best["Candidat"]
        )

    with col2:
        st.metric(
            "TF-IDF",
            f'{best["TF-IDF"]} %'
        )

    with col3:
        st.metric(
            "SBERT",
            f'{best["SBERT"]} %'
        )

    with col4:
        st.metric(
            "Score final",
            f'{best["Score"]} %'
        )


    # --------------------------------
    # BARRE DE PROGRESSION
    # --------------------------------

    st.progress(
        min(best["Score"] / 100, 1.0)
    )


    # --------------------------------
    # COMPETENCES
    # --------------------------------

    st.subheader(
        "Compétences détectées"
    )

    if best["Compétences"]:

        st.write(
            best["Compétences"]
        )

    else:

        st.warning(
            "Aucune compétence détectée."
        )


    # --------------------------------
    # GRAPHIQUE
    # --------------------------------

    st.header("📊 Scores des candidats")

    chart = results[
        [
            "Candidat",
            "TF-IDF",
            "SBERT",
            "Score"
        ]
    ].set_index("Candidat")

    st.bar_chart(chart)