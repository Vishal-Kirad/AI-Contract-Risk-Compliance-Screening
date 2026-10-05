import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from pypdf import PdfReader

# ============================================================
# 1. Load saved model artifacts
# ============================================================

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("final_tfidf.pkl")
    model = joblib.load("final_model.pkl")
    threshold = joblib.load("risk_threshold.pkl")
    return tfidf, model, threshold

tfidf, model, threshold = load_artifacts()


# ============================================================
# 2. Text cleaning
# ============================================================

def clean_contract_text(text):
    text = text.lower()
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[._-]{4,}", " ", text)
    return text.strip()


# ============================================================
# 3. Prediction
# ============================================================

def predict_contract(text):
    cleaned_text = clean_contract_text(text)
    vector = tfidf.transform([cleaned_text])

    probabilities = model.predict_proba(vector)[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = model.classes_[predicted_index]
    confidence = float(np.max(probabilities))

    decision = "Automatic" if confidence >= threshold else "Manual Review"

    return predicted_class, confidence, decision, probabilities, vector


# ============================================================
# 4. PDF text extraction
# ============================================================

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


# ============================================================
# 5. Page configuration
# ============================================================

st.set_page_config(
    page_title="Contract Risk Dashboard",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Contract Risk Dashboard")
st.caption("AI-assisted contract risk screening")

st.info(
    "This dashboard is for risk screening and decision support only. "
    "It is not legal advice."
)


# ============================================================
# 6. Session history
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []


# ============================================================
# 7. Sidebar
# ============================================================

st.sidebar.header("Settings")
st.sidebar.write(f"Review threshold: **{threshold:.2f}**")
st.sidebar.write("Below threshold → Manual Review")


# ============================================================
# 8. Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a contract PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Analyze Contract", type="primary"):

        with st.spinner("Analyzing contract..."):

            try:
                text = extract_pdf_text(uploaded_file)

                if not text.strip():
                    st.error(
                        "No selectable text found in this PDF. "
                        "Scanned/image-only PDFs are not supported by this simple version."
                    )
                    st.stop()

                prediction, confidence, decision, probabilities, vector = (
                    predict_contract(text)
                )

                # Save result in session history
                record = {
                    "Document": uploaded_file.name,
                    "Risk": prediction,
                    "Confidence": confidence,
                    "Decision": decision
                }

                # Avoid duplicate consecutive entries
                if not st.session_state.documents or (
                    st.session_state.documents[-1]["Document"]
                    != uploaded_file.name
                ):
                    st.session_state.documents.append(record)

                # ------------------------------------------------
                # Result
                # ------------------------------------------------

                st.subheader("Risk Assessment")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Risk Level",
                        prediction
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}"
                    )

                with col3:
                    st.metric(
                        "Decision",
                        decision
                    )

                # Flag message
                if decision == "Manual Review":
                    st.warning(
                        "⚠️ FLAGGED: Confidence is below the review threshold. "
                        "This document should be reviewed manually."
                    )
                else:
                    st.success(
                        "✅ High-confidence prediction. "
                        "No manual review is triggered by the confidence rule."
                    )

                # ------------------------------------------------
                # Probabilities
                # ------------------------------------------------

                st.subheader("Class Probabilities")

                probability_df = pd.DataFrame({
                    "Risk Class": model.classes_,
                    "Probability": probabilities
                })

                probability_df["Probability"] = (
                    probability_df["Probability"] * 100
                ).round(2)

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # Top ML supporting features
                # ------------------------------------------------

                class_index = list(model.classes_).index(prediction)
                coefficients = model.coef_[class_index]
                feature_names = tfidf.get_feature_names_out()

                feature_indices = vector.nonzero()[1]
                feature_values = (
                    vector[0, feature_indices].toarray().ravel()
                )

                contributions = (
                    feature_values *
                    coefficients[feature_indices]
                )

                explanation_df = pd.DataFrame({
                    "Feature": feature_names[feature_indices],
                    "Contribution": contributions
                })

                explanation_df = explanation_df[
                    explanation_df["Contribution"] > 0
                ].sort_values(
                    "Contribution",
                    ascending=False
                ).head(10)

                st.subheader("Top Supporting Features")

                if not explanation_df.empty:
                    explanation_df["Contribution"] = (
                        explanation_df["Contribution"].round(5)
                    )
                    st.dataframe(
                        explanation_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.write("No strong supporting features found.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ============================================================
# 9. Flagged Documents
# ============================================================

st.divider()
st.subheader("🚩 Flagged / Reviewed Documents")

flagged_documents = [
    doc for doc in st.session_state.documents
    if doc["Decision"] == "Manual Review"
]

if flagged_documents:

    flagged_df = pd.DataFrame(flagged_documents)

    flagged_df["Confidence"] = (
        flagged_df["Confidence"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        flagged_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.success("No documents are currently flagged for manual review.")


# ============================================================
# 10. All analyzed documents
# ============================================================

if st.session_state.documents:

    st.subheader("All Analyzed Documents")

    all_df = pd.DataFrame(st.session_state.documents)

    all_df["Confidence"] = (
        all_df["Confidence"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        all_df,
        use_container_width=True,
        hide_index=True
    )
