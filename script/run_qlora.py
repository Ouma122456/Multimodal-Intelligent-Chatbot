# run_qlora.py
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# =========================
# 1. Model
# =========================
model_name = "models/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading model in 4-bit with QLoRA...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# prepare for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# =========================
# 2. LoRA Configuration
# =========================
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# print trainable params
model.print_trainable_parameters()

# =========================
# 3. Load Pre-tokenized Dataset
# =========================
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/tokenized/train_tokenized.jsonl",
        "validation": "data/tokenized/val_tokenized.jsonl",
    },
    field=None,
)

def prepare_lm(example):
    example["input_ids"] = example["tokens"]
    example["labels"] = example["tokens"]
    return example

dataset = dataset.map(prepare_lm)
dataset.set_format(type="torch", columns=["input_ids", "labels"])

# =========================
# 4. Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir="./lora_llama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="epoch",
    eval_strategy="epoch",  # 👈 use eval_strategy for your Transformers version
    fp16=False,
    bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
    optim="paged_adamw_32bit",
    report_to="none",
)

# =========================
# 5. Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args
)

# =========================
# 6. Train
# =========================
trainer.train()

# Save LoRA adapter
trainer.model.save_pretrained("./lora_adapter")
print("✅ Training complete! LoRA adapter saved in ./lora_adapter")
