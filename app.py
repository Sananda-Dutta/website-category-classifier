import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("best_website_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Page config
st.set_page_config(page_title="Website Classifier", layout="centered")

st.title("🌐 Website Category Classifier")
st.markdown("### AI-powered Multi-class Text Classification System")

st.write("Enter website content text to predict its category.")

# Display model accuracy
MODEL_ACCURACY = 0.93  # Update with your tuned accuracy
st.info(f"📊 Model Test Accuracy: {MODEL_ACCURACY*100:.2f}%")

# Input box
user_input = st.text_area("Enter Website Text", height=150)

if st.button("Predict Category"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        # Prediction
        prediction = model.predict([user_input])
        probabilities = model.predict_proba([user_input])[0]

        predicted_category = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)

        st.success(f"🏷 Predicted Category: {predicted_category}")
        st.write(f"🔎 Confidence Score: {confidence*100:.2f}%")

        # Show probability distribution
        st.markdown("### 📊 Probability Distribution")
        prob_dict = {
            label_encoder.inverse_transform([i])[0]: prob
            for i, prob in enumerate(probabilities)
        }

        st.bar_chart(prob_dict)

# Optional dropdown display
st.markdown("---")
st.markdown("### 📂 Available Categories")

categories = label_encoder.classes_
selected = st.selectbox("Browse Categories", categories)
st.write(f"You selected: **{selected}**")
