🩺 AutoDoctor — AI Health Assistant

AutoDoctor is a full-stack AI healthcare assistant that predicts possible diseases from user symptoms and medical images (Chest X-rays), and recommends nearby hospitals based on pincode.

⚠️ Disclaimer: This project is for educational purposes only and not a medical diagnosis system.

⸻

🚀 Features

📝 Symptom-based Disease Prediction
	•	Uses NLP (TF-IDF + Logistic Regression)
	•	Accepts free-text symptoms
	•	Returns top 3 possible diseases with confidence scores

🩻 Chest X-ray Image Classification
	•	Uses CNN (ResNet-18)
	•	Detects:
	•	Normal
	•	Pneumonia
	•	Image upload supported via frontend
Brain and abdomen tumor predictor.
🏥 Hospital Recommendation
	•	Finds nearby hospitals using:
	•	Indian hospital dataset (CSV)
	•	OpenStreetMap (fallback)
	•	Google Maps links included

🌐 Full-Stack App
	•	Backend: FastAPI
	•	Frontend: Streamlit
	•	ML: Scikit-learn + PyTorch
🧠 Machine Learning Models

1️⃣ Symptom Model
	•	Algorithm: Logistic Regression
	•	Vectorizer: TF-IDF
	•	Input: Symptom text
	•	Output: Disease probabilities

2️⃣ Image Model
	•	Architecture: ResNet-18
	•	Framework: PyTorch
	•	Input: Chest X-ray image
	•	Output: Normal / Pneumonia
