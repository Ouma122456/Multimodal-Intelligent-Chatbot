# test2.py
import streamlit as st
from llama_cpp import Llama
import speech_recognition as sr
import re

# ✅ Proper import with package style
import personality 
from rag import setup_rag, rag_query,chat_memory
import faiss
import numpy as np

# ------------------ LOAD MODELS ------------------
# Load LLaMA model once
llm = Llama(model_path=r"C:\Users\abass\llama.cpp\models\llama-pro-8b-instruct.Q4_K_M.gguf")
personality.llama_model = llm  # inject into personality.py

# Setup RAG (load embeddings, FAISS, LLM)
setup_rag()

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="LLaMA + RAG Chat", layout="centered")
st.title("💬 LLaMA + RAG Chat Interface")

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "running_mic" not in st.session_state:
    st.session_state["running_mic"] = False

# ------------------ DOCUMENT UPLOAD ------------------
uploaded_file = st.file_uploader(
    "📄 Upload a document (PDF, TXT, DOCX) for RAG to use", 
    type=["pdf", "txt", "docx"]
)

if uploaded_file:
    from document_loader import load_document  # your loader
    from rag import chunk_text, embed_model, index, all_chunks  # reuse chunking & FAISS
    
    try:
        text = load_document(uploaded_file)
        st.success(f"Document '{uploaded_file.name}' loaded successfully!")

        # 👉 Afficher le contenu brut du document
        st.subheader("📖 Extracted Document Content:")
        st.write(text[:2000])  # limit display for very large docs

        # Index the document for RAG
        all_chunks.extend(chunk_text(text))
        embeddings = embed_model.encode(all_chunks, convert_to_tensor=True).cpu().numpy()
        index.reset()
        index.add(embeddings)
        st.info("✅ Document indexed! You can now ask questions about it.")
    except Exception as e:
        st.error(f"❌ Error while processing document: {e}")

# ------------------ MICROPHONE SECTION ------------------
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for background noise... please wait")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        st.info("Listening... Speak now 👂")
        recognizer.pause_threshold = 2
        audio = recognizer.listen(source, timeout=None)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError:
        st.error("Could not request results; check your internet connection")
        return None

# ------------------ MICROPHONE BUTTONS ------------------
mic_col1, mic_col2 = st.columns([1, 1])

with mic_col1:
    if st.button("🎤 Start Microphone"):
        st.session_state.running_mic = True
        spoken_text = record_audio()
        if spoken_text:
            st.session_state["messages"].append({"role": "user", "content": spoken_text})
            with st.spinner("Generating response..."):
                # RAG query
                rag_answer = rag_query(spoken_text)
                if rag_answer == "Not found in document":
                    bot_reply = rag_query(prompt)

                else:
                    personality_answer = personality.generate_response(spoken_text)
                    bot_reply = f"{rag_answer}\n\n(Note: {personality_answer})"

                # Clean non-ASCII chars
                bot_reply = re.sub(r"[^\x00-\x7F]+", "", bot_reply)

                st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
        st.session_state.running_mic = False

with mic_col2:
    st.button("🛑 Stop Microphone", on_click=lambda: st.session_state.update({"running_mic": False}))

# ------------------ CHAT INTERFACE ------------------
prompt = st.chat_input("Ask me something...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.spinner("Generating response..."):
        rag_answer = rag_query(prompt)
        if rag_answer == "Not found in document":
            bot_reply = rag_query(prompt)

        else:
            personality_answer = personality.generate_response(prompt)
            bot_reply = f"{rag_answer}\n\n(Note: {personality_answer})"

        # Clean non-ASCII chars
        bot_reply = re.sub(r"[^\x00-\x7F]+", "", bot_reply)
        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
