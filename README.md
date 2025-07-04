# Job-Role Classifier 🚀  
*Turn any résumé or job-description into a job-family label in < 2 s.*

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn)](#)
[![Streamlit Ready](https://img.shields.io/badge/Streamlit-UI_ready-brightgreen?logo=streamlit)](#)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> **94 % macro-F1 (ensemble TF-IDF → VotingClassifier) on the Kaggle “Updated Resume Dataset”.**  
> Saves a production-ready pickle (`job_role_pipeline.pkl`) plus its label encoder—plug them straight into a Streamlit or FastAPI app.

---

## 1  Project Overview
Recruiters lose countless hours skimming résumés.  
This project trains an NLP ensemble that classifies raw résumé / JD text into **25 categories** (Data Scientist, DevOps Engineer, …) with high accuracy, then serialises the model for real-time inference.

**Pipeline**

1. 🗂  **Dataset** Download & unzip *UpdatedResumeDataSet.csv* from Kaggle.  
2. 🧹  **Cleaning** Remove URLs, emails, punctuation + English stop-words.  
3. 🔡  **Vectorise** `TfidfVectorizer()` (unigram+bigram).  
4. 🤖  **Model** Soft-voting ensemble of LR, Random Forest, SVC, XGBoost, Multinomial NB, Decision Tree.  
5. 🔍  **Tuning** `GridSearchCV` over key hyper-params (3-fold).  
6. 📈  **Evaluation** Accuracy, macro-F1, class report.  
7. 💾  **Export** Save best pipeline + `LabelEncoder` with `joblib`.  

---

## 2  Project Structure
├─ project_1.py ← training script (Colab-generated)
├─ resume-dataset/ ← auto-extracted CSV
├─ job_role_pipeline.pkl ← ⬅️ trained model (generated)
├─ label_encoder.pkl ← ⬅️ classes mapping (generated)
├─ requirements.txt
└─ README.md


---

## 3  Quick Start


# 1. Clone
git clone https://github.com/<you>/job-role-classifier.git
cd job-role-classifier

# 2. Create + activate env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Export Kaggle creds (⚠️ do NOT hard-code them!)
export KAGGLE_USERNAME="your_id"
export KAGGLE_KEY="xxxxxxxxxxxxxxxxxxxx"

# 5. Train & export model
python project_1.py
