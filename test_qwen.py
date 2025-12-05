from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to where we just downloaded the model
model_path = "./models/Qwen2.5-7B-Instruct"

print("Loading Tokenizer...")
# trust_remote_code is needed for Qwen
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading Model onto GPUs (Auto-splitting)...")
# device_map="auto" will automatically use your V100s
# torch_dtype=torch.float16 reduces memory usage by half (crucial for 7B)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print(f"Model loaded! Device Map: {model.hf_device_map}")

# Let's try a simple generation
prompt = "Tokyo is a city known for"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Formatting the input
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating response...")
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=100
)
# Decode the output (turn numbers back into words)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 30)
print(f"Prompt: {prompt}")
print(f"Qwen Response: {response}")
print("-" * 30)