# 🤖 Design of a Multimodal Intelligent Chatbot

This repository contains the implementation of a **multimodal intelligent chatbot** developed during my internship at **EDEV**, a Tunisian company specialized in e-commerce solutions.  
The project demonstrates how to design a **secure, local, and customizable assistant** powered by modern AI techniques such as **Large Language Models (LLMs)**, **speech processing**, and **Retrieval-Augmented Generation (RAG)**.

---

## 🧠 Project Overview

The main goal of this project was to build an intelligent chatbot capable of interacting via **text and voice** while ensuring **data privacy** by running entirely within EDEV’s internal environment.  
It leverages the **LLaMA Pro 8B Instruct** model and was deployed through a **Streamlit interface** for intuitive use.

### 🎯 Objectives
- Develop a multimodal chatbot supporting **text and voice** input.  
- Implement **contextual memory** for coherent multi-turn conversations.  
- Add **personality and sentiment analysis** to adapt to user emotions.  
- Explore **fine-tuning (LoRA/QLoRA)** for model specialization.  
- Integrate **RAG (Retrieval-Augmented Generation)** for document-based responses.  

---

## ⚙️ Tech Stack

| Component | Description |
|------------|-------------|
| **Programming Language** | Python |
| **Frameworks** | Streamlit, Transformers, llama-cpp-python |
| **AI/ML Libraries** | PyTorch, SentenceTransformers, FAISS |
| **Speech Processing** | speechrecognition (Speech-to-Text) |
| **Data Format** | JSONL for training and evaluation |
| **Version Control** | Git, GitHub |
| **Environment** | Windows 11, VS Code, Conda |

---

## 🚀 Features

- **Text and Voice Interaction:** Natural, multimodal communication.
- **Contextual Memory:** Maintains conversation flow and context.
- **Sentiment Analysis:** Adapts tone based on user emotion.
- **Customizable Personality:** Adjust chatbot behavior dynamically.
- **RAG Integration:** Retrieves and uses relevant information from documents.
- **Local Execution:** Ensures data confidentiality (no cloud dependency).

---

## 🧪 Implementation Highlights

- Deployed the chatbot through a **Streamlit web interface**.
- Integrated **speech recognition** for multimodal interaction.
- Added **context and personality modules** for realistic dialogue.
- Conducted **fine-tuning experiments** using LoRA/QLoRA.
- Tested **RAG pipelines** for document-based knowledge retrieval.

---

## 🧱 Challenges & Solutions

| Challenge | Solution |
|------------|-----------|
| High computational cost for fine-tuning | Used LoRA/QLoRA to reduce resource usage |
| Library conflicts (llama-cpp-python) | Migrated to a Streamlit-based setup |
| Limited hardware | Optimized model size and quantization |
| RAG integration issues | Implemented a prototype pipeline for proof of concept |

---

## 📈 Results & Future Work

The project successfully produced a functional multimodal chatbot capable of:
- Handling both **text and voice** inputs,
- Maintaining conversation history,
- Performing basic **sentiment adaptation**,
- Demonstrating the feasibility of **local LLM deployment**.

### Future Improvements:
- Complete fine-tuning on high-performance infrastructure.
- Expand RAG for full document indexing and retrieval.
- Add **text-to-speech** for natural spoken responses.
- Develop a **web API** for broader system integration.

---



## 👩‍💻 Author

**Oumayma ABASSI**  
Data Engineering and Decision Systems | ENET’Com Sfax  
📧 [abassioumayma22@gmail.com](mailto:abassioumayma22@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/oumayma-abassi-a6697)  

---

## 📚 Keywords
`AI` `Chatbot` `LLM` `Multimodality` `Fine-Tuning` `RAG` `Streamlit` `Speech-to-Text`

---

© 2025 Oumayma Abassi. All rights reserved.


