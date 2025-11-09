from llama_cpp import Llama
import json
from pathlib import Path

# Load GGUF model
model_path = Path("models/llama-pro-8b-instruct.Q4_K_M.gguf")
llm = Llama(model_path=str(model_path))

# Paths to tokenized dataset
train_file = Path("data/tokenized/train_tokenized.jsonl")
val_file = Path("data/tokenized/val_tokenized.jsonl")

# Load tokenized dataset
def load_tokenized(file_path):
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

train_data = load_tokenized(train_file)
val_data = load_tokenized(val_file)

# Example usage: feed a tokenized example to the model
# Note: llama_cpp does not have built-in HF-style Trainer
for ex in train_data[:5]:
    input_ids = ex["tokens"]  # from your tokenized dataset
    # You could use llm.generate or a custom training loop here
    print("Example tokens:", input_ids)
