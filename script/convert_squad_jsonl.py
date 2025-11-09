import json

def convert_squad_jsonl(input_file, output_file):
    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            context = item["context"]
            question = item["question"]
            answers = item["answers"]["text"]

            if not answers:  # skip if no answer
                continue

            answer = answers[0]  # use first answer

            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            completion = " " + answer  # leading space for OpenAI fine-tuning best practice

            data.append({"prompt": prompt, "completion": completion})

    # Save back to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Converted {input_file} → {output_file} with {len(data)} samples")


# Convert train & validation separately
convert_squad_jsonl("train.jsonl", "train_prepared.jsonl")
convert_squad_jsonl("validation.jsonl", "validation_prepared.jsonl")
