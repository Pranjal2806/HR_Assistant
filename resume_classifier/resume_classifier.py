import streamlit as st
import pickle
import docx  
import PyPDF2  
import re
import os
import pandas as pd

def load_model_files():
    """Load the pre-trained model, TF-IDF vectorizer, and label encoder."""
    try:
        base_path = os.path.dirname(__file__)
        knn_model = pickle.load(open(os.path.join(base_path, 'clf.pkl'), 'rb'))
        tfidf = pickle.load(open(os.path.join(base_path, 'tfidf.pkl'), 'rb'))
        label_encoder = pickle.load(open(os.path.join(base_path, 'encoder.pkl'), 'rb'))
        return knn_model, tfidf, label_encoder
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

# Load the model, vectorizer, and encoder
knn_model, tfidf, label_encoder = load_model_files()

def clean_resume_text(text):
    """Clean the resume text by removing unwanted characters and patterns."""
    text = re.sub(r'http\S+\s', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+\s', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_uploaded_file(file):
    extension = file.name.split('.')[-1].lower()
    if extension == 'pdf':
        return extract_text_from_pdf(file)
    elif extension == 'docx':
        return extract_text_from_docx(file)
    elif extension == 'txt':
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def predict_resume_category(text):
    cleaned_text = clean_resume_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    dense_vectorized_text = vectorized_text.toarray()
    predicted_label = knn_model.predict(dense_vectorized_text)
    return label_encoder.inverse_transform(predicted_label)[0]

def run_resume_classifier():
    """Main function to run the Streamlit app for multiple resumes."""
    # ❌ Removed st.set_page_config (must be in main_app.py only)

    st.title("Resume Quality Classifier")
    st.markdown("Upload one or more resumes (PDF, DOCX, or TXT format) to classify their TYPE.")

    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    results = []

    if uploaded_files:
        for file in uploaded_files:
            st.divider()
            st.subheader(f"📄 {file.name}")

            try:
                resume_text = handle_uploaded_file(file)
                st.success("Resume text extracted successfully.")

                if st.checkbox(f"Show Extracted Text - {file.name}", value=False):
                    st.text_area(f"Extracted Text ({file.name})", resume_text, height=300)

                predicted_category = predict_resume_category(resume_text)
                st.markdown(f"###  Role: **{predicted_category}**")

                results.append({"Filename": file.name, "Prediction": predicted_category})

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

        if results:
            st.divider()
            st.subheader("📊 Summary of Predictions")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='resume_predictions.csv',
                mime='text/csv',
            )
