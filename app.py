import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -----------------------------
# Paths (robust to working dir)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "features.pkl")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "data", "precautions.csv")

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# Load ML model + features
# -----------------------------
@st.cache_resource
def load_model_and_features():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    return model, feature_names

@st.cache_data
def load_disease_info():
    return pd.read_csv(DISEASE_INFO_PATH)

try:
    model, feature_names = load_model_and_features()
except FileNotFoundError as e:
    st.error(f"Model files not found. Run train_model.py first.\n\n{e}")
    st.stop()

try:
    disease_info = load_disease_info()
except FileNotFoundError as e:
    st.error(f"Disease info CSV not found at {DISEASE_INFO_PATH}.\n\n{e}")
    st.stop()

# -----------------------------
# Initialize LLM (lazy + safe)
# -----------------------------
llm = None
if GROQ_API_KEY:
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")

st.title("🩺 AI Healthcare Assistant")
st.markdown("Predict diseases + Chat with AI doctor")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This AI-powered healthcare assistant predicts a likely condition "
        "from the symptoms you select, and lets you ask follow-up questions "
        "to an AI chatbot."
    )

    st.divider()

    st.subheader("How it works")
    st.markdown(
        "1. Select your symptoms\n"
        "2. Click **Predict Disease**\n"
        "3. Review precautions & diet tips\n"
        "4. Ask the chatbot follow-up questions"
    )

    st.divider()

    st.subheader("Chatbot status")
    if GROQ_API_KEY:
        st.success("✅ AI chatbot is active")
    else:
        st.error("⚠️ GROQ_API_KEY not set — chatbot is disabled. Add it to a .env file.")

    st.divider()

    st.warning("⚠️ Not a real doctor. Always consult a licensed professional for medical advice.")

# -----------------------------
# Symptom selection
# -----------------------------
st.subheader("🔍 Select Symptoms")

selected_symptoms = st.multiselect(
    "Search and select symptoms",
    options=sorted(feature_names),
    format_func=lambda x: x.replace("_", " ").title()
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
        st.stop()

    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in feature_names]
    # Use a DataFrame with matching column names to avoid sklearn feature-name warnings
    input_data = pd.DataFrame([input_vector], columns=feature_names)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    severity = len(selected_symptoms)
    if severity <= 3:
        severity_level = "🟢 Mild"
    elif severity <= 6:
        severity_level = "🟠 Moderate"
    else:
        severity_level = "🔴 Severe"

    st.session_state["prediction"] = prediction
    st.session_state["confidence"] = float(max(proba) * 100)
    st.session_state["symptoms"] = selected_symptoms
    st.session_state["severity"] = severity_level

    st.subheader("🧾 Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Disease", prediction)
    c2.metric("Confidence", f"{max(proba)*100:.2f}%")
    st.progress(float(max(proba)))
    st.caption(f"Confidence: {max(proba)*100:.2f}%")
    c3.metric("Severity", severity_level)

    # Chart
    df_chart = pd.DataFrame({
        "Disease": model.classes_,
        "Probability": proba
    }).sort_values("Probability", ascending=False).head(5)

    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(df_chart.set_index("Disease"))

    df_chart_display = df_chart.copy()
    df_chart_display["Probability"] *= 100
    st.dataframe(df_chart_display, hide_index=True, use_container_width=True)

    # Precautions
    st.subheader("💊 Disease Information")

    disease_row = disease_info[
        disease_info["Disease"].str.lower() == str(prediction).lower()
    ]

    if not disease_row.empty:
        st.success(f"🦠 Disease: {prediction}")

        st.markdown("### 📖 Description")
        st.write(disease_row.iloc[0]["Description"])

        st.markdown("### ✅ Precautions")
        st.write(disease_row.iloc[0]["Precautions"])

        st.markdown("### 🥗 Recommended Diet")
        st.write(disease_row.iloc[0]["Diet"])

        st.markdown("### 👨‍⚕️ Doctor Recommendation")
        st.write(disease_row.iloc[0]["Doctor Recommendation"])
    else:
        st.warning("No information available for this disease.")

# ---------------- CHATBOT SECTION ---------------- #

st.divider()
st.subheader("💬 AI Health Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

def send_message():
    user_input = st.session_state.chat_input
    if not user_input:
        return

    if llm is None:
        st.session_state.messages.append(f"🧑: {user_input}")
        st.session_state.messages.append("🤖: Chatbot is disabled because GROQ_API_KEY is not set.")
        st.session_state.chat_input = ""
        return

    st.session_state.messages.append(f"🧑: {user_input}")

    prediction = st.session_state.get("prediction", "No prediction available")
    confidence = st.session_state.get("confidence", 0)
    symptoms = st.session_state.get("symptoms", [])

    prompt = f"""
        You are an AI Healthcare Assistant.

        The ML model predicted:

        Disease: {prediction}

        Confidence: {confidence:.2f}%

        Selected symptoms:
        {", ".join(symptoms)}

        Rules:
        - Explain the predicted disease in simple language.
        - Suggest general precautions.
        - Do NOT prescribe medicines.
        - Recommend consulting a doctor if symptoms are severe.
        - This is educational information only.

        User question:
        {user_input}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        reply = response.content
    except Exception as e:
        reply = f"⚠️ Error contacting the AI model: {e}"

    st.session_state.messages.append(f"🤖: {reply}")
    st.session_state.chat_input = ""  # clears the box on next render

# Display chat history
for msg in st.session_state.messages:
    st.write(msg)

# Input box — Enter key or button both trigger send_message via on_change/callback
st.text_input("Ask a health question:", key="chat_input", on_change=send_message)
st.button("Send", on_click=send_message)
