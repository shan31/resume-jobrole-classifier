import streamlit as st
import joblib
import PyPDF2
import numpy as np
import matplotlib.pyplot as plt
import shap

# Load the full pipeline and label encoder
model = joblib.load("job_role_pipeline.pkl")  # ✅ Includes TF-IDF + VotingClassifier
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Job Role Classifier", layout="centered")
st.title("💼 Job Role Classifier")
st.write("Upload a resume (PDF) or paste a job description to predict the job category.")

option = st.radio("Choose input method:", ["Upload Resume (PDF)", "Paste Text"])

text_input = ""

if option == "Upload Resume (PDF)":
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text_input = "".join(page.extract_text() or "" for page in pdf_reader.pages)

elif option == "Paste Text":
    text_input = st.text_area("Paste the job description or resume text below:")

if st.button("Predict Job Role") and text_input.strip():
    try:
        input_text = text_input.strip()

        # 1️⃣ Predict and decode
        raw_pred = model.predict([input_text])
        prediction = int(raw_pred[0])
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🧠 Predicted Job Role: **{predicted_label}**")

        # 2️⃣ Vectorize
        tfidf = model.named_steps["tfidf"]
        input_vec = tfidf.transform([input_text])
        input_arr = input_vec.toarray()          # dense 1×N_features
        feature_names = tfidf.get_feature_names_out()

        # 3️⃣ Get the XGBoost sub-model
        xgb_model = model.named_steps["clf"].estimators_[3]  # 'xgb' in your VotingClassifier

        # 4️⃣ SHAP TreeExplainer (no interactions, minimal memory)
        explainer = shap.TreeExplainer(
            xgb_model,
            feature_perturbation="interventional"  # lighter than default
        )
        # compute shap values for our single row
        shap_vals = explainer.shap_values(input_arr, check_additivity=False)

        # 5️⃣ Extract the 1D array of SHAP scores
        if isinstance(shap_vals, list):
            # multi-class: each entry is array (n_samples, n_features)
            shap_scores = shap_vals[prediction][0]
        else:
            # binary/regression
            shap_scores = shap_vals[0]

        # 6️⃣ Pick top 5 words
        top_idx = np.argsort(-np.abs(shap_scores))[:5]
        top_words  = [feature_names[i] for i in top_idx]
        top_scores = shap_scores[top_idx]

        # 7️⃣ Plot horizontal bar chart
        # st.subheader("🔍 Top 5 Words That Influenced the Prediction (XGBoost SHAP)")
        fig, ax = plt.subplots()
        ax.barh(top_words[::-1], top_scores[::-1], color="skyblue")
        ax.set_xlabel("SHAP value")
        ax.set_title("Feature importance (local)")
        st.pyplot(fig)

    except Exception as e:
        # Fallback: use xgb internal importances
        # … inside your except block, before plotting …
        try:
            xgb_model = model.named_steps["clf"].estimators_[3]
            # get raw feature importances (as 'f1234': score)
            fmap = xgb_model.get_booster().get_score(importance_type="weight")

            # get the TF-IDF vocabulary (feature names)
            tfidf = model.named_steps["tfidf"]
            feature_names = tfidf.get_feature_names_out()

            # pick top 5 (f### keys sorted by score)
            items = sorted(fmap.items(), key=lambda kv: kv[1], reverse=True)[:5]

            # convert 'f5722' → 5722 → feature_names[5722]
            top_words  = [feature_names[int(fname[1:])] for fname, _ in items]
            top_scores = [score for _, score in items]

            # st.warning("⚠️ SHAP failed, showing XGBoost feature_importances_ instead")
            fig, ax = plt.subplots()
            ax.barh(top_words[::-1], top_scores[::-1], color="salmon")
            ax.set_xlabel("Importance (weight)")
            ax.set_title("Global feature importances")
            st.pyplot(fig)

        except Exception as fallback_e:
            st.error(f"❌ Error during fallback: {fallback_e}")
