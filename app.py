
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

import plotly.express as px
import fitz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


# =====================================================
# 1. CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Assistant Intelligent de Recrutement",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Recrutement AI")
st.markdown(
    "Analysez les CV et identifiez les profils les plus pertinents "
    "grâce à **TF-IDF** et **Sentence-BERT**."
)


# =====================================================
# 2. CHARGEMENT DU MODELE
# =====================================================
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# 3. EXTRACTION DU TEXTE DES PDF
# =====================================================
def extract_pdf_text(pdf_file):
    try:
        text_parts = []

        with fitz.open(
            stream=pdf_file.getvalue(),
            filetype="pdf"
        ) as document:
            for page in document:
                text_parts.append(page.get_text())

        return "\n".join(text_parts).strip()

    except Exception as error:
        st.warning(
            f"Erreur de lecture du fichier "
            f"{pdf_file.name} : {error}"
        )
        return ""


def clean_text(text):
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_candidate_name(filename):
    return Path(filename).stem.replace("_", " ").replace("-", " ")


# =====================================================
# 4. PARAMETRES DU MATCHING
# =====================================================
st.sidebar.header("⚙️ Paramètres du matching")

weight_tfidf = st.sidebar.slider(
    "Poids TF-IDF",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

weight_sbert = round(1.0 - weight_tfidf, 1)

st.sidebar.caption(
    f"Poids Sentence-BERT : {weight_sbert:.1f}"
)

st.sidebar.markdown("---")

st.sidebar.header("🏆 Shortlist")

shortlist_size = st.sidebar.selectbox(
    "Nombre de candidats à retenir",
    options=[3, 5, 10, 15],
    index=1
)


# =====================================================
# 5. CREATION DE L'OFFRE D'EMPLOI
# =====================================================
st.subheader("📝 Fiche de poste")

nom_offre = st.text_input(
    "Nom de l'offre d'emploi",
    placeholder="Ex. : Data Analyst"
)

job_description = st.text_area(
    "Description de l'offre",
    placeholder=(
        "Décrivez les missions, les compétences requises, "
        "les qualifications et l'expérience recherchées..."
    ),
    height=180
)


# =====================================================
# 6. IMPORTATION DES CV
# =====================================================
st.subheader("📄 CV des candidats")

uploaded_files = st.file_uploader(
    "Importez les CV au format PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} fichier(s) importé(s).")
else:
    st.caption("Importez un ou plusieurs CV PDF pour commencer.")


# =====================================================
# 7. LANCEMENT DU MATCHING
# =====================================================
if st.button("🚀 Lancer le matching", type="primary"):

    if not nom_offre.strip():
        st.warning("Veuillez saisir le nom de l'offre d'emploi.")

    elif not job_description.strip():
        st.warning("Veuillez saisir la description de l'offre.")

    elif not uploaded_files:
        st.warning("Veuillez importer au moins un CV PDF.")

    else:

        # ---------------------------------------------
        # EXTRACTION DES CV
        # ---------------------------------------------
        candidates = []

        with st.spinner("Extraction du contenu des CV..."):

            for pdf_file in uploaded_files:

                cv_text = clean_text(
                    extract_pdf_text(pdf_file)
                )

                if cv_text:
                    candidates.append({
                        "Nom": get_candidate_name(pdf_file.name),
                        "Fichier": pdf_file.name,
                        "Texte_CV": cv_text
                    })
                else:
                    st.warning(
                        f"Aucun texte exploitable dans "
                        f"{pdf_file.name}. "
                        "Vérifiez le PDF ou utilisez un OCR "
                        "si le document est scanné."
                    )

        if not candidates:
            st.error(
                "Aucun CV exploitable. "
                "Veuillez vérifier les fichiers importés."
            )
            st.stop()

        df = pd.DataFrame(candidates)

        # ---------------------------------------------
        # CALCUL DES SCORES
        # ---------------------------------------------
        with st.spinner(
            "Calcul des scores TF-IDF et Sentence-BERT..."
        ):

            documents = (
                [job_description.strip()]
                + df["Texte_CV"].tolist()
            )

            # TF-IDF
            try:
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )

                tfidf_matrix = vectorizer.fit_transform(
                    documents
                )

                scores_tfidf = cosine_similarity(
                    tfidf_matrix[0:1],
                    tfidf_matrix[1:]
                ).flatten()

            except ValueError:
                scores_tfidf = np.zeros(len(df))

            # Sentence-BERT
            try:
                model = load_sbert_model()

                job_embedding = model.encode(
                    job_description.strip(),
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                cv_embeddings = model.encode(
                    df["Texte_CV"].tolist(),
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                scores_sbert = util.cos_sim(
                    job_embedding,
                    cv_embeddings
                ).cpu().numpy().flatten()

                # Convention d'affichage : score entre 0 et 100.
                scores_sbert = np.clip(
                    scores_sbert, 0, 1
                )

            except Exception as error:
                st.error(
                    f"Erreur Sentence-BERT : {error}"
                )
                st.stop()

            # -----------------------------------------
            # SCORE HYBRIDE
            # -----------------------------------------
            df["Score TF-IDF (%)"] = np.round(
                scores_tfidf * 100, 2
            )

            df["Score SBERT (%)"] = np.round(
                scores_sbert * 100, 2
            )

            df["Score hybride (%)"] = np.round(
                weight_tfidf * df["Score TF-IDF (%)"]
                + weight_sbert * df["Score SBERT (%)"],
                2
            )

            # Classement du meilleur au moins bien classé
            df = df.sort_values(
                by="Score hybride (%)",
                ascending=False
            ).reset_index(drop=True)

            df.insert(
                0,
                "Rang",
                range(1, len(df) + 1)
            )

        # =================================================
        # 8. RESULTATS GENERAUX
        # =================================================
        st.success("Analyse des CV terminée.")

        st.markdown(f"## 📌 Offre analysée : {nom_offre}")
        st.write(job_description)

        st.markdown("---")
        st.subheader("📊 Indicateurs du recrutement")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "CV analysés",
            len(df)
        )

        col2.metric(
            "Meilleur score",
            f"{df.iloc[0]['Score hybride (%)']:.2f} %"
        )

        col3.metric(
            "Candidat en tête",
            df.iloc[0]["Nom"]
        )

        # =================================================
        # 9. TABLEAU DE CLASSEMENT
        # =================================================
        st.markdown("---")
        st.subheader("🏆 Classement des candidats")

        display_columns = [
            "Rang",
            "Nom",
            "Fichier",
            "Score TF-IDF (%)",
            "Score SBERT (%)",
            "Score hybride (%)"
        ]

        st.dataframe(
            df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # 10. GRAPHIQUE DES SCORES
        # =================================================
        st.markdown("---")
        st.subheader("📈 Graphique des scores de matching")

        st.caption(
            "Comparaison des scores hybrides des dix premiers candidats."
        )

        graph_df = df.head(10).copy()

        graph_df = graph_df.sort_values(
            by="Score hybride (%)",
            ascending=True
        )

        fig = px.bar(
            graph_df,
            x="Score hybride (%)",
            y="Nom",
            orientation="h",
            text="Score hybride (%)",
            color="Score hybride (%)",
            color_continuous_scale="Blues",
            range_x=[0, 100],
            labels={
                "Score hybride (%)": "Score de matching (%)",
                "Nom": "Candidat"
            },
            title=f"Matching des candidats — {nom_offre}"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            xaxis_title="Score hybride (%)",
            yaxis_title="Candidat",
            coloraxis_showscale=False,
            height=max(400, len(graph_df) * 50),
            margin=dict(l=20, r=60, t=70, b=30)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # 11. SHORTLIST DES MEILLEURS CANDIDATS
        # =================================================
        st.markdown("---")

        nb_retenus = min(shortlist_size, len(df))

        st.subheader(
            f"⭐ Shortlist des {nb_retenus} meilleurs candidats"
        )

        shortlist = df.head(shortlist_size).copy()

        for _, candidate in shortlist.iterrows():

            with st.expander(
                f"#{int(candidate['Rang'])} — "
                f"{candidate['Nom']} | "
                f"{candidate['Score hybride (%)']:.2f} %"
            ):

                st.write(
                    f"**Nom du fichier :** {candidate['Fichier']}"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "TF-IDF",
                    f"{candidate['Score TF-IDF (%)']:.2f} %"
                )

                c2.metric(
                    "Sentence-BERT",
                    f"{candidate['Score SBERT (%)']:.2f} %"
                )

                c3.metric(
                    "Score hybride",
                    f"{candidate['Score hybride (%)']:.2f} %"
                )

                st.write("**Extrait du CV :**")

                extrait = candidate["Texte_CV"][:1500]

                st.write(
                    extrait + (
                        "..." if len(candidate["Texte_CV"]) > 1500
                        else ""
                    )
                )

        # =================================================
        # 12. EXPORTATION DES RESULTATS
        # =================================================
        st.markdown("---")
        st.subheader("📥 Télécharger les résultats")

        export_columns = [
            "Rang",
            "Nom",
            "Fichier",
            "Score TF-IDF (%)",
            "Score SBERT (%)",
            "Score hybride (%)"
        ]

        col1, col2 = st.columns(2)

        with col1:
            csv_all = df[export_columns].to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "📥 Télécharger tous les résultats",
                data=csv_all,
                file_name="resultats_matching.csv",
                mime="text/csv"
            )

        with col2:
            csv_shortlist = shortlist[
                export_columns
            ].to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "⭐ Télécharger la shortlist",
                data=csv_shortlist,
                file_name="shortlist_candidats.csv",
                mime="text/csv"
            )

        st.caption(
            "Les scores représentent une similarité textuelle, "
            "pas une probabilité d'embauche. Les résultats doivent "
            "être examinés par un recruteur."
        )