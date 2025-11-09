from llama_cpp import Llama

llm = Llama(model_path=r"C:\Users\abass\llama.cpp\models\llama-pro-8b-instruct.Q4_K_M.gguf")

output = llm("Hello, how are you?", max_tokens=100)
print(output["choices"][0]["text"])
