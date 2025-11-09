class ConversationMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.turns = []

    def add_turn(self, user_input, bot_response):
        self.turns.append((user_input, bot_response))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_context(self):
        return "\n".join([f"User: {u}\nBot: {b}" for u, b in self.turns])
