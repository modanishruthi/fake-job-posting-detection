import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Page config
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #6366f1;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
    }
    .result-fake {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        border: 1px solid #ef4444;
        text-align: center;
    }
    .result-real {
        background: linear-gradient(135deg, #052e16, #14532d);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        border: 1px solid #22c55e;
        text-align: center;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-prob {
        font-size: 3rem;
        font-weight: 900;
    }
    .tip-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #f59e0b;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #d1d5db;
    }
    .section-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6366f1;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background-color: #1e2130 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] button:hover {
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# Layout
left_col, right_col = st.columns([1, 2], gap="large")

# LEFT SIDEBAR COLUMN
with left_col:
    st.markdown('<div class="hero-title">🔍 Fake Job<br>Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Powered by Logistic Regression + TF-IDF</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Model Performance")

    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Recall (Fake Class)</div>
        <div class="stat-value">87%</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Overall Accuracy</div>
        <div class="stat-value">96%</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">F1 Score (Fake)</div>
        <div class="stat-value">0.71</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Dataset</div>
        <div class="stat-value">17,880 jobs</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚙️ How It Works")
    st.markdown("""
    <div style="color: #9ca3af; font-size: 0.88rem; line-height: 1.7;">
    1. Your input is combined into a single text<br>
    2. TF-IDF converts it to numeric features<br>
    3. Logistic Regression predicts fake/real<br>
    4. Confidence score shown via predict_proba
    </div>
    """, unsafe_allow_html=True)

# RIGHT INPUT COLUMN
with right_col:
    st.markdown("#### 📝 Enter Job Details")

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g. Data Analyst")
        location = st.text_input("Location", placeholder="e.g. Remote / Hyderabad")
    with col2:
        company = st.text_input("Company Name", placeholder="e.g. Infosys")
        requirements = st.text_input("Requirements (brief)", placeholder="e.g. Python, SQL, 2 years exp")

    description = st.text_area("Job Description", height=130, placeholder="Paste the full job description here...")
    benefits = st.text_area("Benefits", height=80, placeholder="e.g. Health insurance, remote work, stock options")

    analyze = st.button("🔍 Analyze Job Posting")

    if analyze:
        combined = f"{job_title} {company} {location} {description} {requirements} {benefits}"

        if combined.strip():
            transformed = vectorizer.transform([combined])
            prediction = model.predict(transformed)[0]
            probability = model.predict_proba(transformed)[0]

            fake_prob = round(probability[1] * 100, 1)
            real_prob = round(probability[0] * 100, 1)

            st.markdown("---")

            if prediction == 1:
                st.markdown(f"""
                <div class="result-fake">
                    <div class="result-title">⚠️ Fake Job Detected</div>
                    <div class="result-prob" style="color:#f87171;">{fake_prob}%</div>
                    <div style="color:#fca5a5; font-size:0.9rem;">confidence it's fake</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="tip-box">
                    ⚠️ <strong>Red flags to watch for:</strong> Vague job descriptions, unrealistic salaries,
                    requests for personal/banking info, no company details, "no experience needed" claims.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-real">
                    <div class="result-title">✅ Looks Genuine</div>
                    <div class="result-prob" style="color:#4ade80;">{real_prob}%</div>
                    <div style="color:#86efac; font-size:0.9rem;">confidence it's real</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("🟢 Real Probability", f"{real_prob}%")
            c2.metric("🔴 Fake Probability", f"{fake_prob}%")
            st.progress(fake_prob / 100)

        else:
            st.warning("Please fill in at least one field.")