import torch
from sentence_transformers import SentenceTransformer, util
import time

# Import our knowledge base
from tokyo_data import knowledge_base

def main():
    # 1. Setup Device (Let's use GPU 1 as verified)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"--- Running on {device} ---")

    # 2. Load the Model directly onto the GPU
    # 'all-MiniLM-L6-v2' is the standard efficient embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # 3. Vectorization (The "Index" step)
    # We convert ALL our text data into vectors (numbers).
    # Since model is on GPU, this calculation happens on V100.
    print("Embedding knowledge base...")
    start_time = time.time()
    
    # convert_to_tensor=True keeps the data on the GPU memory for speed
    corpus_embeddings = model.encode(knowledge_base, convert_to_tensor=True)
    
    end_time = time.time()
    print(f"Done. Embedded {len(knowledge_base)} sentences in {end_time - start_time:.4f} seconds.")

    # ---------------------------------------------------------
    # 4. The Loop: Ask questions
    # ---------------------------------------------------------
    queries = [
        "What color is the Chuo Line?",
        "How do I go to Narita Airport?",
        "Tell me about the circle line in Tokyo.",
        "Where can I buy old books?"
    ]

    print("\n--- Starting Retrieval Test ---\n")

    for query in queries:
        # A. Embed the query (User Question -> Vector)
        query_embedding = model.encode(query, convert_to_tensor=True)

        # B. Search (Math: Cosine Similarity)
        # We calculate the angle between the Query Vector and ALL Knowledge Vectors.
        # This is a matrix multiplication under the hood.
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

        # C. Get Top 3 results
        top_results = torch.topk(cos_scores, k=3)

        print(f"Q: {query}")
        
        for score, idx in zip(top_results.values, top_results.indices):
            # idx is the index in our knowledge_base list
            print(f"  [Score: {score:.4f}] {knowledge_base[idx]}")
        
        print("-" * 40)

if __name__ == "__main__":
    main()