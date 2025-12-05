import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from pypdf import PdfReader
import os
import argparse

# ==========================================
# 0. Argument Parsing
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG Chatbot for PDF documents (v2.0)")
    parser.add_argument(
        "filename", 
        type=str, 
        help="The path to the PDF file you want to chat with"
    )
    return parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================
args = parse_arguments()
FILE_PATH = args.filename

# [IMPROVEMENT 1] Bigger chunks to capture full definitions/math contexts
CHUNK_SIZE = 1000         
OVERLAP = 200             

if not os.path.exists(FILE_PATH):
    print(f"ERROR: File '{FILE_PATH}' not found!")
    exit(1)

print(f"--- Initializing PDF RAG System v2.0 with {FILE_PATH} ---")

# ==========================================
# 1. Document Loader & Chunker
# ==========================================
def load_and_chunk_pdf(path):
    print(f"Reading {path}...")
    try:
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

    print(f"Total characters read: {len(full_text)}")
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunk = full_text[start:end]
        chunks.append(chunk)
        # Move forward with overlap
        start += (CHUNK_SIZE - OVERLAP)
    
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# Run the loader
knowledge_base = load_and_chunk_pdf(FILE_PATH)

if not knowledge_base:
    print("CRITICAL ERROR: No text extracted.")
    exit(1)

# ==========================================
# 2. Indexing (Embedding)
# ==========================================
print("Loading Retriever and Indexing Chunks...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = embed_model.encode(knowledge_base, convert_to_tensor=True)

# ==========================================
# 3. Load Generator (Qwen)
# ==========================================
print("Loading Generator (Qwen2.5-7B)...")
model_path = "./models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("\n" + "="*50)
print(f" READY! Ask me questions about {FILE_PATH}")
print(" Type 'exit', 'quit', or 'q' to stop.")
print("="*50 + "\n")

# ==========================================
# 4. Chat Loop
# ==========================================
while True:
    try:
        user_input = input("User> ")
    except EOFError:
        break
    
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break
    
    if not user_input.strip():
        continue

    # A. Retrieve
    question_embedding = embed_model.encode(user_input, convert_to_tensor=True)
    
    # [IMPROVEMENT 2] Increase top_k to 5 to get more context
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=5)
    
    # Extract the chunk text
    relevant_chunks = [knowledge_base[hit['corpus_id']] for hit in hits[0]]

    # [IMPROVEMENT 3] THE ABSTRACT TRICK
    # Always insert Chunk 0 (Title/Abstract) at the beginning if it's not already there.
    # This helps Qwen answer general questions like "What is this paper about?"
    if knowledge_base[0] not in relevant_chunks:
        relevant_chunks.insert(0, knowledge_base[0])
        # Mark it in debug print so you know it was forced
        forced_chunk_added = True
    else:
        forced_chunk_added = False

    context_str = "\n---\n".join(relevant_chunks)
    
    # [DEBUG] Print what the bot is reading
    print("\n   [DEBUG: Bot Context Content]")
    if forced_chunk_added:
        print(f"   > Chunk 0 (FORCED): {knowledge_base[0].replace(chr(10), ' ')[:80]}...")
    
    for i, hit in enumerate(hits[0]):
        chunk_id = hit['corpus_id']
        content_preview = knowledge_base[chunk_id].replace('\n', ' ')[:80]
        print(f"   > Chunk {chunk_id} (Score {hit['score']:.4f}): {content_preview}...")
    print("   [End of Debug]\n")

    # B. Prompt
    prompt = f"""You are a Researcher Assistant. 
Answer the question based ONLY on the context below.
If the answer is not in the context, say "I cannot find the answer in the document."

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
    
    # C. Generate
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=250 
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Bot> {response}\n")