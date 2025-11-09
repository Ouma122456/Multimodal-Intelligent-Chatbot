from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

model_name = "meta-llama/Llama-2-7b-hf"

print("Loading model on GPU with 8-bit quantization + CPU offload...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                 # or use load_in_4bit=True for more savings
    llm_int8_enable_fp32_cpu_offload=True  # allow offloading extra layers to CPU
)

# Load with automatic device map + quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = "Hello, who are you?"
output = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])
