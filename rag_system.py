import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. Setup Knowledge Base (The "Library")
# ==========================================
knowledge_base = [
    "The Chuo Line (Rapid) connects Tokyo Station and Takao Station.",
    "The Chuo Line trains are orange in color (E233 series).",
    "The Yamanote Line is a loop line connecting major city centers in Tokyo.",
    "Yamanote Line trains are light green (E235 series).",
    "The Sobu Line (Rapid) runs from Tokyo to Chiba and Narita Airport.",
    "The Sobu Line trains are blue and cream colored.",
    "Kanda Station is famous for used bookstores and curry."
]

print("--- Initializing KandaEnv RAG System ---")

# ==========================================
# 2. Load Retriever (The "Eyes")
# ==========================================
print("[1/3] Loading Embedding Model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# Convert knowledge base to vectors
corpus_embeddings = embed_model.encode(knowledge_base, convert_to_tensor=True)
print("Knowledge base indexed.")

# ==========================================
# 3. Load Generator (The "Brain")
# ==========================================
print("[2/3] Loading Qwen Model (This takes a moment)...")
model_path = "./models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ==========================================
# 4. Define the RAG Function
# ==========================================
def ask_rag(question):
    print(f"\nUser Question: {question}")
    
    # Step A: Retrieve (Find relevant info)
    question_embedding = embed_model.encode(question, convert_to_tensor=True)
    # Search for top 2 most relevant sentences
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=2)
    
    # Extract the text from hits
    relevant_docs = [knowledge_base[hit['corpus_id']] for hit in hits[0]]
    context_str = "\n".join(relevant_docs)
    
    print(f"Retrieved Context:\n{context_str}")
    
    # Step B: Augment (Build the Prompt)
    # We tell Qwen: "Use the context below to answer."
    prompt = f"""You are a Tokyo Railway Guide. Answer the question based ONLY on the following context.
    
Context:
{context_str}

Question: 
{question}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Step C: Generate (Qwen speaks)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=100
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"RAG Answer: {response}")
    print("-" * 50)

# ==========================================
# 5. Run Tests
# ==========================================
print("\n[3/3] System Ready. Starting Test...\n")

# Test 1: Asking about color (Specific fact)
ask_rag("What is the color of the Yamanote Line?")

# Test 2: Semantic question (Concept "Circle" -> "Loop")
ask_rag("Tell me about the circle line.")

# Test 3: Complex routing
ask_rag("How can I go to the airport?")