import streamlit as st
import os
import re
from rag_utility import process_file, ask_question

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Multi-File AI Chatbot", layout="centered")

st.title("Multi-File AI Chatbot")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# Session State
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# RESET BUTTON (IMPORTANT)
# -------------------------------
if st.button("Reset Chat"):
    st.session_state.history = []

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload file",
    type=["pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg"]
)

file_path = None

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded!")

    if st.button("Process File"):
        process_file(file_path)
        st.success("File processed ✅")

# -------------------------------
# Ask Question
# -------------------------------
query = st.chat_input("Ask your question...")

def clean_text(text):
    # ✅ Remove any accidental HTML tags
    return re.sub(r"<.*?>", "", text)

if query:
    answer = ask_question(query)

    clean_query = clean_text(query)
    clean_answer = clean_text(answer)

    st.session_state.history.append((clean_query, clean_answer))


# -------------------------------
# Chat Display (BEST METHOD)
# -------------------------------
st.markdown("### Chat")

for q, a in st.session_state.history:

    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        st.markdown(a)
