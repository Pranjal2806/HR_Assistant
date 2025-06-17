# main_app.py

import streamlit as st

# ========== Import All Three Modules ==========
from resume_classifier.resume_classifier import run_resume_classifier
from sustainability_checker.sustainability_checker import run_sustainability_checker
from qa_module.qa_module import run_qa_module

# ========== Set Page Config ==========
st.set_page_config(page_title="Resume Intelligence Suite", page_icon="🧠", layout="wide")

st.title("🧠 Resume Intelligence Suite")
st.markdown("Select a module from the sidebar to get started.")

# ========== Sidebar for Module Selection ==========
module = st.sidebar.radio(
    "Select Module:",
    ("Resume Classifier", "Sustainability Checker", "Q&A Generator")
)

# ========== Render Selected Module ==========
if module == "Resume Classifier":
    run_resume_classifier()

elif module == "Sustainability Checker":
    run_sustainability_checker()

elif module == "Q&A Generator":
    run_qa_module()
