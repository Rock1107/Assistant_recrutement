import streamlit as st
import pandas as pd
from outils.pdf_parser import extract_pdf_text
from outils.Extraction import extract_skills
from outils.Matching import calculate_matching_score
# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Assistant intelligent de recrutement",
    page_icon="🤖",
    layout="wide"
)
# ============================================
# TITRE
# ============================================
st.title("🤖 Recrutement AI")
st.write(
    "Analyse automatique des CV et mise en correspondance "
    "avec une offre d'emploi grâce au TF-IDF et à SBERT."
)
# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
@st.cache_data
def load_data():
    offres = pd.read_csv(
        "offres_emploi_synthetiques.csv",
        encoding="utf-8-sig"
    )
    candidatures = pd.read_csv(
        "candidatures_synthetiques.csv",
        encoding="utf-8-sig"
    )
    return offres, candidatures
# ============================================
# CHOIX DU MODE
# ============================================
st.sidebar.header("⚙️ Mode de fonctionnement")
mode = st.sidebar.radio(
    "Choisissez le mode :",
    [
        "📊 Données synthétiques",
        "📄 Importer des CV PDF"
    ]
)
# ============================================
# MODE DONNÉES SYNTHÉTIQUES
# ============================================
if mode == "📊 Données synthétiques":
    try:
        offres, candidatures = load_data()
        st.success(
            "✅ Données synthétiques chargées avec succès."
        )
        # ----------------------------------------
        # STATISTIQUES
        # ----------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "📋 Nombre d'offres",
                len(offres)
            )
        with col2:
            st.metric(
                "👥 Nombre de candidats",
                len(candidatures)
            )
        # ----------------------------------------
        # SÉLECTION DE L'OFFRE
        # ----------------------------------------
        st.header("1️⃣ Sélection de l'offre d'emploi")
        offre_selectionnee = st.selectbox(
            "Choisissez une offre :",
            range(len(offres)),
            format_func=lambda i: (
                f"{offres.iloc[i]['id_offre']} - "
                f"{offres.iloc[i]['poste']}"
            )
        )
        offre = offres.iloc[offre_selectionnee]
        st.subheader(
            f"💼 {offre['poste']}"
        )
        st.write(
            f"**Description :** {offre['description']}"
        )
        st.write(
            f"**Compétences requises :** "
            f"{offre['competences']}"
        )
        st.write(
            f"**Niveau d'études :** "
            f"{offre['niveau_etudes']}"
        )
        st.write(
            f"**Expérience requise :** "
            f"{offre['experience_requise']}"
        )
        # ----------------------------------------
        # LANCEMENT DU MATCHING
        # ----------------------------------------
        if st.button(
            "🚀 Lancer le matching",
            type="primary"
        ):
            job_description = (
                f"{offre['poste']}. "
                f"{offre['description']}. "
                f"Compétences requises : "
                f"{offre['competences']}. "
                f"Niveau d'études : "
                f"{offre['niveau_etudes']}. "
                f"Expérience : "
                f"{offre['experience_requise']}."
            )
            cv_texts = (
                candidatures["cv_text"]
                .fillna("")
                .tolist()
            )
            # ------------------------------------
            # MATCHING TF-IDF + SBERT
            # ------------------------------------
            (
                tfidf_scores,
                sbert_scores,
                final_scores
            ) = calculate_matching_score(
                job_description,
                cv_texts
            )
            # ------------------------------------
            # CONSTRUCTION DES RÉSULTATS
            # ------------------------------------
            results = candidatures.copy()
            results["TF-IDF"] = [
                round(float(score), 2)
                for score in tfidf_scores
            ]
            results["SBERT"] = [
                round(float(score), 2)
                for score in sbert_scores
            ]
            results["Score"] = [
                round(float(score), 2)
                for score in final_scores
            ]
            # ------------------------------------
            # CLASSEMENT
            # ------------------------------------
            results = (
                results
                .sort_values(
                    by="Score",
                    ascending=False
                )
                .reset_index(drop=True)
            )
            # ------------------------------------
            # RÉSULTATS
            # ------------------------------------
            st.success(
                "✅ Analyse terminée avec succès."
            )
            st.header(
                "3️⃣ Classement des candidats"
            )
            display_results = results[
                [
                    "candidate_id",
                    "nom",
                    "poste_cible",
                    "competences",
                    "TF-IDF",
                    "SBERT",
                    "Score"
                ]
            ].copy()
            display_results.columns = [
                "ID",
                "Candidat",
                "Poste cible",
                "Compétences",
                "TF-IDF",
                "SBERT",
                "Score final"
            ]
            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True
            )
            # ------------------------------------
            # MEILLEUR CANDIDAT
            # ------------------------------------
            best = results.iloc[0]
            st.header(
                "🏆 Meilleur candidat"
            )
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Candidat",
                    best["nom"]
                )
            with col2:
                st.metric(
                    "TF-IDF",
                    f"{best['TF-IDF']} %"
                )
            with col3:
                st.metric(
                    "SBERT",
                    f"{best['SBERT']} %"
                )
            with col4:
                st.metric(
                    "Score final",
                    f"{best['Score']} %"
                )
            st.progress(
                min(float(best["Score"]) / 100, 1.0)
            )
            # ------------------------------------
            # PROFIL DU MEILLEUR CANDIDAT
            # ------------------------------------
            st.subheader(
                "👤 Profil du meilleur candidat"
            )
            st.write(
                f"**Nom :** {best['nom']}"
            )
            st.write(
                f"**Diplôme :** {best['diplome']}"
            )
            st.write(
                f"**Expérience :** {best['experience']}"
            )
            st.write(
                f"**Certifications :** "
                f"{best['certifications']}"
            )
            st.write(
                f"**Compétences :** "
                f"{best['competences']}"
            )
            # ------------------------------------
            # GRAPHIQUE
            # ------------------------------------
            st.header(
                "📊 Comparaison des scores"
            )
            chart = results[
                [
                    "nom",
                    "TF-IDF",
                    "SBERT",
                    "Score"
                ]
            ].set_index("nom")
            st.bar_chart(chart)
    except Exception as e:
        st.error(
            f"❌ Une erreur est survenue : {e}"
        )
# ============================================
# MODE IMPORT PDF
# ============================================
else:
    st.header("1️⃣ Offre d'emploi")
    job_title = st.text_input(
        "Intitulé du poste",
        placeholder="Exemple : Data Analyst"
    )
    job_description = st.text_area(
        "Description de l'offre",
        height=200,
        placeholder=(
            "Exemple : Nous recherchons un Data Analyst "
            "maîtrisant Python, SQL, Power BI et Excel."
        )
    )
    # ----------------------------------------
    # IMPORT DES CV
    # ----------------------------------------
    st.header("2️⃣ CV des candidats")
    uploaded_files = st.file_uploader(
        "Importer les CV",
        type=["pdf"],
        accept_multiple_files=True
    )
    # ----------------------------------------
    # ANALYSE
    # ----------------------------------------
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
        # ------------------------------------
        # EXTRACTION DES CV
        # ------------------------------------
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
        # ------------------------------------
        # MATCHING
        # ------------------------------------
        (
            tfidf_scores,
            sbert_scores,
            final_scores
        ) = calculate_matching_score(
            job_description,
            cv_texts
        )
        # ------------------------------------
        # AJOUT DES SCORES
        # ------------------------------------
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
        # ------------------------------------
        # CLASSEMENT
        # ------------------------------------
        candidates = sorted(
            candidates,
            key=lambda x: x["Score"],
            reverse=True
        )
        # ------------------------------------
        # RÉSULTATS
        # ------------------------------------
        st.success(
            "✅ Analyse terminée avec succès."
        )
        st.header(
            "3️⃣ Classement des candidats"
        )
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
        # ------------------------------------
        # MEILLEUR CANDIDAT
        # ------------------------------------
        best = candidates[0]
        st.header(
            "🏆 Meilleur candidat"
        )
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
        st.progress(
            min(best["Score"] / 100, 1.0)
        )
        # ------------------------------------
        # COMPÉTENCES
        # ------------------------------------
        st.subheader(
            "🧠 Compétences détectées"
        )
        if best["Compétences"]:
            st.write(
                best["Compétences"]
            )
        else:
            st.warning(
                "Aucune compétence détectée."
            )
        # ------------------------------------
        # GRAPHIQUE
        # ------------------------------------
        st.header(
            "📊 Scores des candidats"
        )
        chart = results[
            [
                "Candidat",
                "TF-IDF",
                "SBERT",
                "Score"
            ]
        ].set_index("Candidat")
        st.bar_chart(chart)