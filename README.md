Website Classification App

This project is a Machine Learning Website Classification application. It classifies websites into categories based on their content. The project includes data preprocessing, model training, testing, and deployment using Streamlit for a web interface.

Features

End-to-end ML pipeline: EDA → Preprocessing → Training → Testing → Deployment

Interactive web app with Streamlit

Real-time website category prediction

Supports .pkl model and label encoder files

How to Run

Clone the repository:

git clone <your-github-repo-link>
cd website-classification

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Enter the website input in the app and see the predicted category.

Project Files

app.py – Streamlit application

website_classifier.pkl – Trained ML model

label_encoder.pkl – Label encoder for categories

requirements.txt – Python dependencies

Technologies Used

Python

scikit-learn

pandas, numpy

Streamlit

Author

Sananda Dutta
