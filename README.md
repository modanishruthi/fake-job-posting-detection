# JobShield – Fake Job Posting Detection

An AI-powered web application that detects fraudulent job postings to protect job seekers using NLP and Machine Learning.

## Live Demo
[Click here to view the app]
fake-job-posting-detection-gjy6oeowseux7nc2yucmsp.streamlit.app
(#) <!-- add streamlit link after deployment -->

## Features
- Detects fake vs real job postings with confidence score
- TF-IDF based text analysis on job title, description, requirements, and company profile
- Red flag analysis highlighting suspicious patterns
- Dark-themed Streamlit UI with Plotly gauge chart

## Tech Stack
- Python
- Scikit-learn (Logistic Regression)
- Pandas, NumPy
- TF-IDF Vectorization
- Streamlit
- Plotly

## Model Details
- Dataset: EMSCAD Kaggle Dataset (~17,880 job listings)
- Features: TF-IDF (max 5,000 features) on combined text fields
- Models compared: Logistic Regression vs SVM
- Selected: Logistic Regression (higher recall on fake jobs + predict_proba support)

## Author
Shruthi Modani — B.Tech CSE (AI/ML) | VIT-AP University
