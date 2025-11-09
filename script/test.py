import streamlit as st
from llama_cpp import Llama
import speech_recognition as sr
import os

# ------------------ IMPORT YOUR MODULES ------------------
from personality import generate_response, memory, current_personality
import personality

# ------------------ LOAD MODEL ------------------
llm = Llama(model_path=r"C:\Users\abass\llama.cpp\models\llama-pro-8b-instruct.Q4_K_M.gguf")

# Assign model to personality.py
personality.llama_model = llm

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="LLaMA Chat", layout="centered")
st.title("💬 LLaMA Chat Interface")

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "running_mic" not in st.session_state:
    st.session_state["running_mic"] = False

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

# ------------------ PROCESS QUERY ------------------
def process_query(user_text: str) -> str:
    # Use the personality-driven response generator
    return generate_response(user_text)

# ------------------ MICROPHONE UI ------------------
mic_col1, mic_col2 = st.columns([1, 1])

with mic_col1:
    if st.button("🎤 Start Microphone"):
        st.session_state.running_mic = True
        spoken_text = record_audio()
        if spoken_text:
            st.session_state["messages"].append({"role": "user", "content": spoken_text})
            with st.spinner("Generating response..."):
                bot_reply = process_query(spoken_text)
                st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
        st.session_state.running_mic = False

with mic_col2:
    st.button("🛑 Stop Microphone", on_click=lambda: st.session_state.update({"running_mic": False}))

# ------------------ CHAT INTERFACE ------------------
prompt = st.chat_input("Ask me something...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.spinner("Generating response..."):
        bot_reply = process_query(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
