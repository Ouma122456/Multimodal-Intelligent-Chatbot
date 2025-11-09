# personality.py
from memory import ConversationMemory
from sentiment import analyze_tone

PERSONALITIES = {
    "friendly": "You are a very friendly assistant. Always respond warmly with emojis when possible.",
    "professional": "You are a professional assistant. Be concise and polite.",
    "humorous": "You are a witty and humorous assistant. Always add a little joke."
}

current_personality = PERSONALITIES["humorous"]  # default
memory = ConversationMemory(max_turns=5)
llama_model = None  # Will be assigned from main script

def generate_response(user_input: str) -> str:
    if llama_model is None:
        raise ValueError("llama_model not loaded. Assign it in your main script.")
    
    context = memory.get_context()
    tone_instruction = analyze_tone(user_input)
    prompt = f"{tone_instruction}\n{current_personality}\n{context}\nUser: {user_input}\nBot:"

    response = llama_model(prompt, max_tokens=512)["choices"][0]["text"].strip()
    memory.add_turn(user_input, response)
    return response
