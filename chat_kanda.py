import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# ==========================================
# 1. Setup Knowledge Base
# ==========================================
knowledge_base = [
    "The Chuo Line (Rapid) connects Tokyo Station and Takao Station.",
    "The Chuo Line trains are orange in color (E233 series).",
    "The Yamanote Line is a loop line connecting major city centers in Tokyo.",
    "Yamanote Line trains are light green (E235 series).",
    "The Sobu Line (Rapid) runs from Tokyo to Chiba and Narita Airport.",
    "The Sobu Line trains are blue and cream colored.",
    "Kanda Station is famous for used bookstores and curry shops.",
    "Akihabara is the center of anime and electronics culture.",
    "Shinjuku Station is the busiest train station in the world."
]

print("--- Initializing Interactive Kanda Bot (Strict Mode) ---")

# ==========================================
# 2. Load Models (Retriever + Generator)
# ==========================================
print("Loading Retriever (MiniLM)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = embed_model.encode(knowledge_base, convert_to_tensor=True)

print("Loading Generator (Qwen2.5-7B)... This takes about 10-20 seconds...")
model_path = "./models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("\n" + "="*50)
print(" SYSTEM READY. Type 'exit' or 'quit' to stop.")
print(" Ask me anything about Tokyo trains!")
print("="*50 + "\n")

# ==========================================
# 3. Interactive Loop
# ==========================================
while True:
    # Get user input
    try:
        user_input = input("User> ")
    except EOFError:
        break

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break
    
    if not user_input.strip():
        continue

    # Step A: Retrieve (Increased top_k to 4)
    question_embedding = embed_model.encode(user_input, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=4)
    
    relevant_docs = [knowledge_base[hit['corpus_id']] for hit in hits[0]]
    context_str = "\n".join(relevant_docs)
    
    # Debug info
    print(f"   [Debug: Retrieved {len(relevant_docs)} facts]")

    # Step B: Prompt (Strict Constraint)
    prompt = f"""You are a STRICT Tokyo Guide. 
Answer the question based ONLY on the following context. 
If the answer is NOT in the context, say "I don't know based on the provided info."
DO NOT use your internal knowledge.

Context:
{context_str}

Question: 
{user_input}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Step C: Generate
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=150
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"KandaBot> {response}\n")