# 🩺 AI Healthcare Assistant

An AI-powered web app that predicts likely diseases from user-selected symptoms using a machine learning model, and lets users ask natural-language follow-up questions to an AI chatbot for simple, educational explanations.

> ⚠️ **Disclaimer:** This project is for **learning and educational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed doctor for real health concerns.

---

## ✨ Features

- 🔍 **Symptom-based disease prediction** using a trained Random Forest classifier (41 diseases, 132 symptoms)
- 📊 **Prediction confidence & probability breakdown** with an interactive chart of the top likely diseases
- 🟢🟠🔴 **Severity indicator** based on number of symptoms selected
- 💊 **Disease information panel** — description, precautions, recommended diet, and when to see a doctor
- 💬 **AI Health Chatbot** powered by Groq's LLaMA 3.3 (via LangChain) for follow-up questions, grounded in the model's prediction
- 🎨 Clean, responsive **Streamlit** interface

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| ML Model | scikit-learn (`RandomForestClassifier`) |
| Chatbot / LLM | [Groq API](https://groq.com/) (LLaMA 3.3 70B) via [LangChain](https://www.langchain.com/) (`langchain`, `langchain-groq`, `langchain-core`) |
| Data handling | pandas, numpy |
| Config | python-dotenv (for `.env` / `GROQ_API_KEY`) |

---

## 📂 Project Structure

```
AI-Healthcare-Assistant/
│
├── app.py                  # Main Streamlit application
├── train_model.py          # Trains the disease-prediction model
├── requirements.txt        # Python dependencies
├── .env                    # GROQ_API_KEY (not committed)
├── README.md
│
├── data/
│   ├── dataset.csv         # Symptom–disease training data
│   └── precautions.csv     # Disease info: description, precautions, diet, doctor advice
│
└── model/
    ├── model.pkl           # Trained Random Forest model (generated)
    └── features.pkl        # Ordered list of symptom feature names (generated)
```

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-healthcare-assistant.git
cd ai-healthcare-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

> Get a free API key from [console.groq.com](https://console.groq.com/).

### 5. Train the model

Make sure `data/dataset.csv` is in place, then run:

```bash
python train_model.py
```

This generates `model/model.pkl` and `model/features.pkl`.

### 6. Launch the app

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Select symptoms** — choose from a searchable list of 132 symptoms.
2. **Predict Disease** — a Random Forest model trained on labeled symptom data returns the most likely disease along with a confidence score and severity level.
3. **Review information** — description, precautions, diet recommendations, and doctor guidance are pulled for the predicted disease.
4. **Ask the chatbot** — follow-up questions are answered by an LLM, which is given the prediction and selected symptoms as context so its answers stay relevant and safe (no prescriptions, always recommends professional consultation when appropriate).

---

## 📈 Model Notes & Limitations

- The model is trained on a public symptom–disease dataset where each disease is defined by a fairly fixed set of symptoms. Selecting **more symptoms** generally produces higher, more reliable confidence scores; a handful of symptoms shared across several conditions (e.g. stomach pain, acidity) can lead to lower confidence as probability is split across multiple plausible diseases.
- Predictions are for **educational demonstration only** and should never be used for real diagnosis.

---

## 🗺️ Roadmap / Ideas for Improvement

- [ ] Calibrate model probabilities (`CalibratedClassifierCV`) for more realistic confidence scores
- [ ] Add symptom severity/duration as additional input features
- [ ] Show top-3 differential diagnoses by default instead of a single prediction
- [ ] Add persistent chat history per session
- [ ] Deploy to Streamlit Community Cloud / HuggingFace Spaces

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open-sourced for educational purposes. Add a license of your choice (e.g. MIT) if you plan to share it publicly.

---

## 👤 Author

**[Your Name]**
GitHub: _add your GitHub profile link here_
