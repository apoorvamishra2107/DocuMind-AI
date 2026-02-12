import streamlit as st
from rag_engine import process_pdf, query_vector_store
from gemini_handler import get_gemini_response
import os

st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Header */
.header-box {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
}

/* Subtitle */
.subtext {
    text-align: center;
    font-size: 16px;
    margin-top: 6px;
    opacity: 0.9;
}

/* Sidebar gradient */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f1c2c, #928dab);
    color: white;
}

/* File uploader */
section[data-testid="stSidebar"] .stFileUploader {
    background: rgba(255,255,255,0.12);
    padding: 14px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* Browse button */
section[data-testid="stSidebar"] button {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: black;
    border: none;
    font-weight: 600;
}

/* USER message bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 14px;
    backdrop-filter: blur(12px);
}

/* ASSISTANT message bubble */
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: rgba(0,0,0,0.35);
    border-radius: 14px;
    padding: 16px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
}

/* 🔥 FORCE ALL TEXT INSIDE CHAT TO WHITE */
[data-testid="stChatMessage"] * {
    color: white !important;
}

/* Success box */
div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15);
}

div[data-testid="stAlert"] p {
    color: white !important;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<div class="header-box">
    🤖 Enterprise AI Knowledge Assistant
</div>
<div class="subtext">
    Intelligent Document Analysis & Conversational AI Powered by Google Gemini
</div>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- SIDEBAR ----------
with st.sidebar:

    # Only drag & drop uploader (no extra label box)
    uploaded_file = st.file_uploader(
        label="",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        with st.spinner("Processing document..."):
            process_pdf(uploaded_file)
        st.success("Document processed successfully")

# ---------- CHAT ----------
st.divider()

user_query = st.chat_input("Ask anything about your document...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    if os.path.exists("faiss_index"):
        context = query_vector_store(user_query)

        prompt = f"""
You are an enterprise AI assistant.

Use the context below to answer accurately and professionally.

Context:
{context}

Question:
{user_query}
"""
        answer = get_gemini_response(prompt)
    else:
        answer = get_gemini_response(user_query)

    st.session_state.chat_history.append(("assistant", answer))

# ---------- DISPLAY CHAT ----------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
