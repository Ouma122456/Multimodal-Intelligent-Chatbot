# scripts/preprocess.py
import json
from pathlib import Path
from llama_cpp import Llama

# Paths
data_dir = Path("data")
train_file = data_dir / "train.jsonl"
val_file = data_dir / "validation.jsonl"
tokenized_dir = data_dir / "tokenized"
tokenized_dir.mkdir(exist_ok=True, parents=True)

# Load your LLaMA GGUF model
llm = Llama(model_path="models/llama-pro-8b-instruct.Q4_K_M.gguf")

def tokenize_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            data = json.loads(line)   # load JSON per line
            # ✅ use your actual fields if it's "prompt" + "completion"
            text = data.get("text") or (data.get("prompt", "") + data.get("completion", ""))
            
            if not isinstance(text, str):
                text = str(text)
            
            tokens = llm.tokenize(text.encode("utf-8"))  # ✅ fix here
            f_out.write(json.dumps({"tokens": tokens}) + "\n")


# Tokenize train and validation
tokenize_file(train_file, tokenized_dir / "train_tokenized.jsonl")
tokenize_file(val_file, tokenized_dir / "val_tokenized.jsonl")
