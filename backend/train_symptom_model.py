import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HOSPITAL_PATH = os.path.join(BASE_DIR, "..", "data", "india_hospitals.csv")

hospital_df = pd.read_csv(HOSPITAL_PATH)


hospital_df.columns = hospital_df.columns.str.strip().str.lower()

# 🔥 force pincode to string
hospital_df["pincode"] = hospital_df["pincode"].astype(str)

print("Hospital columns:", hospital_df.columns.tolist())
# absolute path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "symptoms.csv")

df = pd.read_csv("data/final_symptoms_to_disease.csv")
df.columns = df.columns.str.strip().str.lower()


X = df["symptom_text"]
y = df["diseases"]


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X, y)

MODEL_PATH = os.path.join(BASE_DIR, "symptom_model.pkl")
joblib.dump(model, MODEL_PATH)

print("✅ Symptom model trained & saved")