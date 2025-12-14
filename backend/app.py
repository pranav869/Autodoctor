from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import joblib
import requests
import pandas as pd
from urllib.parse import quote_plus
import io

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ================= APP =================
app = FastAPI()

# ================= BASE DIR =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= DEVICE =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD SYMPTOM MODEL =================
SYMPTOM_MODEL_PATH = os.path.join(BASE_DIR, "symptom_model.pkl")
symptom_model = joblib.load(SYMPTOM_MODEL_PATH)

# ================= LOAD CNN MODEL =================
CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.pt")

cnn_model = models.resnet18(weights=None)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)  # NORMAL, PNEUMONIA
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
cnn_model = cnn_model.to(DEVICE)
cnn_model.eval()

# ================= CNN TRANSFORM =================
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = ["Normal", "Pneumonia"]

# ================= LOAD HOSPITAL DATA =================
HOSPITAL_PATH = os.path.join(BASE_DIR, "..", "data", "india_hospitals.csv")
hospital_df = pd.read_csv(HOSPITAL_PATH)

hospital_df.columns = hospital_df.columns.str.strip().str.lower()
hospital_df["pincode"] = (
    hospital_df["pincode"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
)

if "unnamed: 0" in hospital_df.columns:
    hospital_df.drop(columns=["unnamed: 0"], inplace=True)

# ================= INPUT SCHEMA =================
class InputData(BaseModel):
    symptoms: str
    pincode: str

# ================= HELPER FUNCTIONS =================
def get_hospitals_by_pincode(pincode: str):
    filtered = hospital_df[hospital_df["pincode"] == str(pincode)]

    if filtered.empty:
        return []

    hospitals = filtered["hospital"].dropna().unique()[:5]
    return [
        {
            "name": h,
            "map": f"https://www.google.com/maps/search/{quote_plus(h)}"
        }
        for h in hospitals
    ]


def get_lat_lon(pincode: str):
    url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&country=India&format=json"
    r = requests.get(url, timeout=5)

    if r.status_code != 200 or not r.text.strip():
        return None, None

    data = r.json()
    if not data:
        return None, None

    return float(data[0]["lat"]), float(data[0]["lon"])


def get_hospitals_from_api(lat, lon):
    query = f"""
    [out:json];
    node["amenity"="hospital"](around:10000,{lat},{lon});
    out;
    """
    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query,
        timeout=10
    )

    if r.status_code != 200:
        return []

    data = r.json()
    results = []

    for el in data.get("elements", [])[:5]:
        name = el.get("tags", {}).get("name")
        if name:
            results.append({
                "name": name,
                "map": f"https://www.google.com/maps/search/{quote_plus(name)}"
            })

    return results

# ================= SYMPTOM PREDICTION =================
@app.post("/predict")
def predict(data: InputData):
    probs = symptom_model.predict_proba([data.symptoms])[0]
    diseases = symptom_model.classes_

    top = sorted(
        zip(diseases, probs),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    hospitals = get_hospitals_by_pincode(data.pincode)

    if not hospitals:
        lat, lon = get_lat_lon(data.pincode)
        hospitals = get_hospitals_from_api(lat, lon) if lat else []

    return {
        "predictions": [
            {"disease": d, "confidence": round(p * 100, 2)}
            for d, p in top
        ],
        "hospitals": hospitals,
        "disclaimer": "This is not a medical diagnosis."
    }

# ================= IMAGE (CNN) PREDICTION =================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = cnn_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = cnn_model(image)   # ✅ FIXED
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return {
            "prediction": CLASS_NAMES[predicted.item()],
            "confidence": round(confidence.item() * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}