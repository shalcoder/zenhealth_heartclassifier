import streamlit as st
import pandas as pd
import joblib
import requests 
# Page config
st.set_page_config(page_title="Cardiac Disease Predictor", layout="wide")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/cardiac_disease_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler = load_model()

# Sidebar
st.sidebar.title("Cardiac Disease Predictor")
st.sidebar.write("Model Accuracy: 90.76%")
st.sidebar.write("Precision: 90.00%")
st.sidebar.write("Recall: 92.86%")

# Main tabs
tab1, tab2 = st.tabs(["Disease Predictor", "Chatbot"])

# TAB 1: PREDICTOR
with tab1:
    st.header("Cardiac Disease Risk Assessment")

    st.subheader("Primary Cardiac Indicators")

    col1, col2 = st.columns(2)

    with col1:
        st_slope = st.selectbox("ST Slope", [0, 1, 2, 3])
        exercise_angina = st.radio("Exercise Angina", [0, 1])
        chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4])

    with col2:
        max_hr = st.number_input("Max Heart Rate (bpm)", 60, 202, 140)
        oldpeak = st.number_input("ST Depression", -2.6, 6.2, 0.6)

    st.subheader("Demographics")

    col3, col4 = st.columns(2)

    with col3:
        sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    with col4:
        age = st.number_input("Age", 28, 77, 54)

    if st.button("Predict"):
        # Create input
        input_df = pd.DataFrame([{
            'ST slope': st_slope,
            'exercise angina': exercise_angina,
            'chest pain type': chest_pain,
            'max heart rate': max_hr,
            'oldpeak': oldpeak,
            'sex': sex,
            'age': age
        }])

        # Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # Results
        st.write("---")
        st.subheader("Results")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            if prediction == 0:
                st.success("No Disease")
            else:
                st.error("Disease Detected")

        with col_r2:
            risk = "Low" if probability[1] < 0.4 else "Medium" if probability[1] < 0.7 else "High"
            st.write(f"Risk Level: {risk}")

        with col_r3:
            st.write(f"Confidence: {max(probability)*100:.1f}%")

        st.write(f"No Disease: {probability[0]*100:.1f}%")
        st.write(f"Disease: {probability[1]*100:.1f}%")

        # Store for chatbot
        st.session_state.prediction = {
            'result': prediction,
            'prob': probability.tolist(),
            'inputs': {
                'ST Slope': st_slope,
                'Exercise Angina': exercise_angina,
                'Chest Pain': chest_pain,
                'Max HR': max_hr,
                'Oldpeak': oldpeak,
                'Sex': sex,
                'Age': age
            }
        }

# TAB 2: CHATBOT (GEMINI HEALTHCARE ASSISTANT)
with tab2:
    st.header("Healthcare Assistant : Ask · Analyse · Act")

    if 'prediction' not in st.session_state:
        st.warning("⚠️ No prediction available. Please complete Tab 1 first.")
    else:
        pred = st.session_state.prediction

        st.subheader("Prediction Summary")
        st.write(f"**Result:** {'No Disease' if pred['result'] == 0 else 'Disease Detected'}")
        st.write(f"**Disease Probability:** {pred['prob'][1] * 100:.1f}%")

        st.error("""
        This assistant explains cardiac results in **simple language**.
        It answers ONLY heart-related questions. If the question is unrelated,
        it will politely decline.
        """)

        question = st.text_area(
            "Enter a question about heart health:",
            placeholder="Example: How can I reduce my heart disease risk?",
        )

        if st.button("Ask Gemini"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    import requests  # ensure requests is available

                    API_KEY = st.secrets["GEMINI_API_KEY"]
                    MODEL_NAME = st.secrets.get("GEMINI_MODEL", "models/gemini-2.5-flash")

                    predicted_text = (
                        "No Cardiac Disease"
                        if pred['result'] == 0 else
                        "Possible Cardiac Disease / Risk Detected"
                    )

                    # SIMPLE LANGUAGE + RESTRICTED TOPIC PROMPT
                    sys_prompt = f"""
You are a heart-health assistant.

Audience:
- Normal people (not doctors, not medical experts)
- They need simple, easy-to-understand language.

Rules:

1. If the question is related to heart or cardiac health:
      • Explain in **simple language**.
      • Use 5–7 short sentences.
      • Avoid medical terminology unless necessary.
      • Give practical and easy suggestions (diet, lifestyle, habit changes).
      • Do NOT mention probabilities, model details, or ST slope, unless asked.

2. If the question is NOT related to cardiac/heart health:
      Reply EXACTLY:
      "I can answer only questions related to heart health."

3. NEVER give a diagnosis. Never sound like a doctor.

Patient info from ML model:
- Prediction result: {predicted_text}
- Probability of cardiac disease: {pred['prob'][1] * 100:.1f}%
- Patient Inputs: {pred['inputs']}
"""

                    data = {
                        "contents": [
                            {"parts": [{"text": sys_prompt + "\nUser Question: " + question}]}
                        ]
                    }

                    URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={API_KEY}"
                    headers = {"Content-Type": "application/json"}

                    with st.spinner("Getting answer from Gemini..."):
                        response = requests.post(URL, headers=headers, json=data)

                    if response.status_code == 200:
                        reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                        st.subheader("Answer")
                        st.write(reply)
                    else:
                        st.error("❌ Gemini API Error: " + response.text)

                except Exception as e:
                    st.error(f"Error: {e}")
