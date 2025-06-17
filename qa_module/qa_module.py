import streamlit as st
import pandas as pd
import os

def run_qa_module():
    # Load the CSV using a safe relative path
    try:
        data_path = os.path.join(os.path.dirname(__file__), "questionDS.csv")
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("❌ Could not find 'questionDS.csv'. Make sure it's in the qa_module/ folder.")
        st.stop()

    data['role'] = data['role'].str.strip().str.lower()  # Normalize roles

    # Function to get random Q&A
    def get_random_qa(role_input, n=10):
        role_input = role_input.strip().lower()
        filtered = data[data['role'] == role_input]
        return filtered.sample(n=min(n, len(filtered))) if not filtered.empty else None

    # Streamlit UI
    st.title("💬 Interview Questions & Answers by Role")
    st.markdown("Select a job role to view relevant interview Q&A.")

    # Role selector
    roles = sorted(data['role'].unique())
    selected_role = st.selectbox("Select a role:", roles)

    # Slider for Q&A count
    num_items = st.slider("How many questions do you want to ask?", 1, 20, 10)

    # Show Q&A
    if st.button("Show Questions and Answers"):
        results_df = get_random_qa(selected_role, num_items)

        if results_df is not None:
            st.subheader(f"📋 Top {len(results_df)} Q&A for: {selected_role.title()}")
            for idx, row in results_df.iterrows():
                st.markdown(f"**{idx + 1}. Q: {row['question']}**")
                st.markdown(f"*A: {row['answer']}*")
                st.markdown("---")
        else:
            st.warning("❌ No data found for that role.")
