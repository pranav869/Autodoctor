import streamlit as st
import requests
from PIL import Image
import io

BACKEND_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="AI Health Assistant",
    layout="centered"
)

st.title("🧠 AI Health Assistant")
st.write("Predict diseases from symptoms or Chest X-ray images")

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📝 Symptom Prediction", "🩻 X-ray Prediction"])

# =====================================================
# TAB 1 — SYMPTOMS
# =====================================================
with tab1:
    st.subheader("Symptom-based Disease Prediction")

    symptoms = st.text_area(
        "Enter symptoms (comma separated)",
        placeholder="fever, cough, headache"
    )

    pincode = st.text_input("Enter your pincode")

    if st.button("Predict Disease"):
        if not symptoms or not pincode:
            st.warning("Please enter symptoms and pincode")
        else:
            try:
                payload = {
                    "symptoms": symptoms,
                    "pincode": pincode
                }

                res = requests.post(f"{BACKEND_URL}/predict", json=payload)
                data = res.json()

                st.subheader("🔍 Possible Diseases")
                for pred in data.get("predictions", []):
                    st.write(f"**{pred['disease']}** → {pred['confidence']}%")

                st.subheader("🏥 Nearby Hospitals")
                hospitals = data.get("hospitals", [])
                if hospitals:
                    for h in hospitals:
                        st.markdown(f"- [{h['name']}]({h['map']})")
                else:
                    st.info("No hospitals found")

                st.info(data.get("disclaimer", ""))

            except Exception as e:
                st.error(f"Backend error: {e}")

# =====================================================
# TAB 2 — X-RAY IMAGE
# =====================================================
with tab2:
    st.subheader("Chest X-ray Prediction (CNN)")

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded X-ray",
            width=500
        )

        if st.button("Predict from Image"):
            try:
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)

                files = {
                    "file": ("xray.jpg", img_bytes, "image/jpeg")
                }

                res = requests.post(
                    f"{BACKEND_URL}/predict-image",
                    files=files
                )

                data = res.json()

                st.subheader("🧠 Prediction Result")

                if "prediction" in data:
                    st.success(f"Result: **{data['prediction']}**")
                    st.write(f"Confidence: **{data['confidence']}%**")
                else:
                    st.error(data.get("error", "Prediction failed"))

            except Exception as e:
                st.error(f"Prediction error: {e}")