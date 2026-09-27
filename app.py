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
        st.success("✅ Données synthétiques chargées avec succès.")

        # --- FONCTION D'AIDE POUR RÉCUPÉRER UNE COLONNE SÉCURISÉE ---
        def get_col(df, options, default=""):
            for opt in options:
                if opt in df.columns:
                    return opt
            return None

        # Noms de colonnes détectés
        id_job_col = get_col(offres, ['id_offre', 'id', 'job_id', 'offre_id', 'ID'])
        poste_col = get_col(offres, ['poste', 'titre', 'job_title', 'intitule'], 'poste')
        desc_col = get_col(offres, ['description', 'desc', 'job_description'], 'description')
        comp_job_col = get_col(offres, ['competences', 'compétences', 'skills', 'competence'], 'competences')
        etudes_col = get_col(offres, ['niveau_etudes', 'niveau_etude', 'etudes', 'diplome'], 'niveau_etudes')
        exp_job_col = get_col(offres, ['experience_requise', 'experience', 'exp_requise'], 'experience_requise')

        # ----------------------------------------
        # STATISTIQUES
        # ----------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📋 Nombre d'offres", len(offres))
        with col2:
            st.metric("👥 Nombre de candidats", len(candidatures))

        # ----------------------------------------
        # SÉLECTION DE L'OFFRE
        # ----------------------------------------
        st.header("1️⃣ Sélection de l'offre d'emploi")

        def format_offre(i):
            poste_val = offres.iloc[i][poste_col] if poste_col else f"Offre {i+1}"
            if id_job_col:
                return f"{offres.iloc[i][id_job_col]} - {poste_val}"
            return f"{i+1} - {poste_val}"

        offre_selectionnee = st.selectbox(
            "Choisissez une offre :",
            range(len(offres)),
            format_func=format_offre
        )

        offre = offres.iloc[offre_selectionnee]

        st.subheader(f"💼 {offre.get(poste_col, 'Offre sélectionnée')}")
        st.write(f"**Description :** {offre.get(desc_col, 'N/A')}")
        st.write(f"**Compétences requises :** {offre.get(comp_job_col, 'N/A')}")
        st.write(f"**Niveau d'études :** {offre.get(etudes_col, 'N/A')}")
        st.write(f"**Expérience requise :** {offre.get(exp_job_col, 'N/A')}")

        # ----------------------------------------
        # LANCEMENT DU MATCHING
        # ----------------------------------------
        if st.button("🚀 Lancer le matching", type="primary"):
            job_description = (
                f"{offre.get(poste_col, '')}. "
                f"{offre.get(desc_col, '')}. "
                f"Compétences requises : {offre.get(comp_job_col, '')}. "
                f"Niveau d'études : {offre.get(etudes_col, '')}. "
                f"Expérience : {offre.get(exp_job_col, '')}."
            )

            # Sécurité pour la colonne cv_text
            cv_col = get_col(candidatures, ['cv_text', 'cv', 'texte_cv', 'text'], 'cv_text')
            cv_texts = candidatures[cv_col].fillna("").tolist() if cv_col else []

            # ------------------------------------
            # MATCHING TF-IDF + SBERT
            # ------------------------------------
            tfidf_scores, sbert_scores, final_scores = calculate_matching_score(
                job_description,
                cv_texts
            )

            # ------------------------------------
            # CONSTRUCTION DES RÉSULTATS
            # ------------------------------------
            results = candidatures.copy()
            results["TF-IDF"] = [round(float(score), 2) for score in tfidf_scores]
            results["SBERT"] = [round(float(score), 2) for score in sbert_scores]
            results["Score"] = [round(float(score), 2) for score in final_scores]

            results = results.sort_values(by="Score", ascending=False).reset_index(drop=True)

            st.success("✅ Analyse terminée avec succès.")
            st.header("3️⃣ Classement des candidats")

            # Noms des colonnes candidats
            cand_id_col = get_col(results, ['candidate_id', 'id', 'ID', 'candidat_id'], 'candidate_id')
            nom_col = get_col(results, ['nom', 'candidat', 'name', 'nom_candidat'], 'nom')
            target_col = get_col(results, ['poste_cible', 'poste', 'target_job'], 'poste_cible')
            comp_cand_col = get_col(results, ['competences', 'compétences', 'skills', 'competence'], 'competences')

            cols_to_display = [c for c in [cand_id_col, nom_col, target_col, comp_cand_col, "TF-IDF", "SBERT", "Score"] if c in results.columns]
            
            st.dataframe(
                results[cols_to_display],
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------
            # MEILLEUR CANDIDAT
            # ------------------------------------
            best = results.iloc[0]
            st.header("🏆 Meilleur candidat")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Candidat", best.get(nom_col, "N/A"))
            with col2:
                st.metric("TF-IDF", f"{best['TF-IDF']} %")
            with col3:
                st.metric("SBERT", f"{best['SBERT']} %")
            with col4:
                st.metric("Score final", f"{best['Score']} %")

            st.progress(min(float(best["Score"]) / 100, 1.0))

            # ------------------------------------
            # PROFIL DU MEILLEUR CANDIDAT
            # ------------------------------------
            diplome_col = get_col(results, ['diplome', 'diplôme', 'degree', 'formation'], 'diplome')
            exp_cand_col = get_col(results, ['experience', 'expérience', 'exp'], 'experience')
            cert_col = get_col(results, ['certifications', 'certification', 'certs'], 'certifications')

            st.subheader("👤 Profil du meilleur candidat")
            st.write(f"**Nom :** {best.get(nom_col, 'N/A')}")
            st.write(f"**Diplôme :** {best.get(diplome_col, 'N/A')}")
            st.write(f"**Expérience :** {best.get(exp_cand_col, 'N/A')}")
            st.write(f"**Certifications :** {best.get(cert_col, 'N/A')}")
            st.write(f"**Compétences :** {best.get(comp_cand_col, 'N/A')}")

            # ------------------------------------
            # GRAPHIQUE
            # ------------------------------------
            st.header("📊 Comparaison des scores")
            if nom_col in results.columns:
                chart = results[[nom_col, "TF-IDF", "SBERT", "Score"]].set_index(nom_col)
                st.bar_chart(chart)

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")
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
                "Competences": ", ".join(skills),
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
                "Competences",
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
                "Aucune competence détectée."
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