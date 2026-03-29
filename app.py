import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Placement Prediction System")

cgpa = st.number_input("Enter CGPA (0-10)", min_value=0.0, max_value=10.0)
iq = st.number_input("Enter IQ (50-200)", min_value=50, max_value=200)

if st.button("Predict"):
    features = np.array([[cgpa, iq]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"Placed ✅ (Confidence: {round(probability*100,2)}%)")
    else:
        st.error(f"Not Placed ❌ (Confidence: {round(probability*100,2)}%)")