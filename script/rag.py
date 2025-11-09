# rag.py
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import random
from memory import ConversationMemory
import memory

print("DEBUG >>> Using memory.py from:", memory.__file__)
print("DEBUG >>> ConversationMemory methods:", dir(ConversationMemory))


# -------- CONFIG --------
DATA_PATH = "data/train.jsonl"
TEXT_FIELD = "prompt"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"
TOP_K = 3
# ------------------------

def natural_reply(fact_type, fact_value):
    if fact_type == "favorite_color":
        options = [
            f"Ah, I remember! {fact_value} is your favorite color. That really suits your style!",
            f"{fact_value}, of course! A lovely choice for your favorite color.",
            f"Right! You told me {fact_value} is your favorite color. It's such a vibrant one!",
            f"I see, {fact_value} is your favorite. Nice taste!"
        ]
    elif fact_type == "favorite_food":
        options = [
            f"Oh yes! {fact_value} is your favorite food. Sounds delicious!",
            f"Yum, {fact_value}! No wonder it’s your favorite.",
            f"Right, you mentioned {fact_value} before. That’s a tasty choice!",
            f"I remember, {fact_value} is the food you enjoy the most. Great pick!"
        ]
    else:
        options = [f"I remember you told me: {fact_value}"]

    return random.choice(options)

# Global variables
embed_model = None
index = None
all_chunks = None
tokenizer = None
model = None
def chunk_text(text, chunk_size=200, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
def setup_rag():
    global embed_model, index, all_chunks, tokenizer, model

    # Only initialize if not done yet
    if embed_model is not None:
        return

    # Check that data exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"RAG dataset not found at {DATA_PATH}")

    # 1. Load embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2. Load dataset
    dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    texts = dataset[TEXT_FIELD]

    # 3. Chunk texts
    

    all_chunks = []
    for text in texts:
        all_chunks.extend(chunk_text(text))

    # 4. Compute embeddings
    embeddings = embed_model.encode(all_chunks, convert_to_tensor=True).cpu().numpy()

    # 5. Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # 6. Load LLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
def natural_reply(fact_type, fact_value):
    if fact_type == "favorite_color":
        options = [
            f"Your favorite color is {fact_value}, right?",
            f"Ah yes, you told me you love {fact_value}.",
            f"Of course, {fact_value} is your color!",
            f"I remember, you said {fact_value} is your favorite."
        ]
    elif fact_type == "favorite_food":
        options = [
            f"You said you like {fact_value}, sounds delicious!",
            f"Of course, {fact_value} is one of your favorites.",
            f"Yup, you told me {fact_value} is your go-to.",
            f"I remember, {fact_value} is the food you enjoy."
        ]
    else:
        options = [f"I remember you told me: {fact_value}"]

    return random.choice(options)

# Create memory object
chat_memory = ConversationMemory(max_turns=5)

def rag_query(query, top_k=TOP_K):
    global embed_model, index, all_chunks, tokenizer, model

    # Make sure models are loaded
    if embed_model is None or index is None or tokenizer is None or model is None:
        setup_rag()

    q_lower = query.lower()  # normalize once

    # --- FACT HANDLING: color ---
    if "my favorite color is" in q_lower:
        color = q_lower.split("my favorite color is")[-1].strip()
        chat_memory.remember_fact("favorite_color", color)
        reply = natural_reply("favorite_color", color)
        chat_memory.add_turn(query, reply)
        return reply

    if "what is my favorite color" in q_lower:
        remembered = chat_memory.recall_fact("favorite_color")
        if remembered:
            reply = natural_reply("favorite_color", remembered)
        else:
            reply = "Hmm, I don’t think you told me yet. What’s your favorite color?"
        chat_memory.add_turn(query, reply)
        return reply

    # --- FACT HANDLING: food ---
    if "i like" in q_lower:
        food = q_lower.replace("i like", "").strip()
        chat_memory.remember_fact("favorite_food", food)
        reply = natural_reply("favorite_food", food)
        chat_memory.add_turn(query, reply)
        return reply

    elif "my favorite food is" in q_lower:
        food = q_lower.replace("my favorite food is", "").strip()
        chat_memory.remember_fact("favorite_food", food)
        reply = natural_reply("favorite_food", food)
        chat_memory.add_turn(query, reply)
        return reply

    if "what is my favorite food" in q_lower:
        remembered = chat_memory.recall_fact("favorite_food")
        if remembered:
            reply = natural_reply("favorite_food", remembered)
        else:
            reply = "Hmm, you haven’t told me your favorite food yet!"
        chat_memory.add_turn(query, reply)
        return reply

    # --- CONTEXT + RAG ---
    if index is None or len(all_chunks) == 0:
        return "⚠️ No document is loaded yet. Please upload a file first."

    q_emb = embed_model.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(q_emb, top_k)
    retrieved_texts = [all_chunks[int(i)] for i in I[0]]

    context = chat_memory.get_context() + "\n".join(retrieved_texts)
    prompt = f"""
    You are an assistant. Answer **only** using the following context.
    If the answer is not in the context, say "Not found in document".

    Context:
    {context}

    User: {query}
    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.7)

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chat_memory.add_turn(query, reply)
    return reply
