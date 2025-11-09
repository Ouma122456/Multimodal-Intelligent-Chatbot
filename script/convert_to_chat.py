import json
from pathlib import Path

data_dir = Path("data")
input_file = data_dir / "validation.jsonl"
output_file = data_dir / "val_chat.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        item = json.loads(line)
        prompt = item["prompt"].strip()
        completion = item["completion"].strip()

        chat_format = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
        }
        f_out.write(json.dumps(chat_format) + "\n")

print(f"✅ Converted {input_file} → {output_file}")
