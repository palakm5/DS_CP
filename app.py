import streamlit as st
from src.predict import predict_crop
import os

# --- Streamlit App Config ---
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# --- App Title ---
st.title("🌾 Intelligent Crop Recommendation System")
st.markdown(
    """
    ### 🌱 Get the Best Crop Suggestion
    Enter your soil and environmental details below — the system will predict the **most suitable crop** using a trained Machine Learning model.
    """
)

# --- Input Fields ---
st.subheader("🧪 Soil Nutrient Levels")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
with col2:
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
with col3:
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)

st.subheader("🌦️ Environmental Conditions")
col4, col5, col6, col7 = st.columns(4)

with col4:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
with col5:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
with col6:
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
with col7:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# --- Predict Button ---
if st.button("🔍 Recommend Crop"):
    try:
        # Call prediction function
        crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"🌱 **Recommended Crop:** {crop.upper()}")
        st.balloons()

    except FileNotFoundError as fnf_error:
        st.error(f"❌ Missing model or scaler file: {fnf_error}")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")

# --- Footer ---
st.markdown(
    """
    ---
    👩‍💻 **Developed by:** *Your Name*  
    💡 *Crop Recommendation using Machine Learning*
    """
)
