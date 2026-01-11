import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Page Setup ---
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🏥")

# --- Load Pre-trained Objects ---
# Ensure karein ke ye files aapne Colab se download karke GitHub pe bhi dali hain
@st.cache_resource
def load_models():
    model = joblib.load('best_diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_transform.pkl')
    return model, scaler, pca

try:
    model, scaler, pca = load_models()
    
    st.title("🏥 Diabetes Prediction System")
    st.markdown("""
    Ye app Patient ke medical data ko analyze karke predict karti hai ke unhe diabetes hone ka khatra hai ya nahi.
    """)
    st.divider()

    # --- User Inputs ---
    st.subheader("Patient ki Details Enter Karein:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        preg = st.number_input("Pregnancies", min_value=0, step=1, value=0)
        glu = st.number_input("Glucose Level", min_value=0, value=120)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=70)
        skin = st.number_input("Skin Thickness (mm)", min_value=0, value=20)

    with col2:
        ins = st.number_input("Insulin (mu U/ml)", min_value=0, value=80)
        bmi = st.number_input("BMI", min_value=0.0, format="%.1f", value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.47)
        age = st.number_input("Age (Years)", min_value=1, step=1, value=30)

    # --- Prediction ---
    if st.button("Predict Result", use_container_width=True):
        # Data ko list mein convert karna
        input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
        
        # 1. Scaling apply karna
        scaled_data = scaler.transform(input_data)
        
        # 2. PCA apply karna (95% variance wala)
        pca_data = pca.transform(scaled_data)
        
        # 3. Final Prediction
        prediction = model.predict(pca_data)
        
        st.divider()
        if prediction[0] == 1:
            st.error("### 🚨 Result: Diabetic (1)")
            st.warning("Model ke mutabiq diabetes ke chances zyada hain. Doctor se ruju karein.")
        else:
            st.success("### ✅ Result: Non-Diabetic (0)")
            st.info("Results normal hain. Patient ko diabetes hone ka khatra kam hai.")

except Exception as e:
    st.error(f"Error loading model files: {e}")
