"""
GenAI Intern Assessment — Conflict-Aware RAG System (NebulaGears)
Author: Ayush (IIT Hyderabad)

Primary LLM      : Google Gemini Flash 2.0
Vector Store     : ChromaDB (local)
Embeddings       : SentenceTransformers (all-MiniLM-L6-v2)
Bonus (Optional) : Local Llama 3.1 8B GGUF fallback
"""

import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

# ----------------------------------------------------
# Load text files (REAL DATA from assignment)
# ----------------------------------------------------
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

docs = {
    "employee_handbook_v1.txt": read_file("employee_handbook_v1.txt"),
    "manager_updates_2024.txt": read_file("manager_updates_2024.txt"),
    "intern_onboarding_faq.txt": read_file("intern_onboarding_faq.txt")
}

# ----------------------------------------------------
# Document Metadata (role + priority)
# ----------------------------------------------------
metadata = {
    "employee_handbook_v1.txt": {"role": "general", "priority": 1},
    "manager_updates_2024.txt": {"role": "general", "priority": 2},
    "intern_onboarding_faq.txt": {"role": "intern", "priority": 10}
}

# ----------------------------------------------------
# Embeddings
# ----------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------------------
# Gemini Setup
# ----------------------------------------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# ----------------------------------------------------
# Optional Llama Fallback
# ----------------------------------------------------
def load_llama():
    try:
        return Llama(
            model_path="models/llama3.gguf",
            n_threads=4,
            n_ctx=8192,        # upgraded to avoid overflow
            n_batch=1024
        )
    except Exception as e:
        print("⚠ Local Llama load failed:", e)
        return None

local_llm = load_llama()

# ----------------------------------------------------
# ChromaDB Setup
# ----------------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma")

collection = chroma_client.get_or_create_collection(
    name="nebulagears_policy",
    metadata={"hnsw:space": "cosine"}
)

# Insert ALL documents properly
for name, text in docs.items():
    emb = embedder.encode(text).tolist()
    collection.upsert(
        ids=[name],
        embeddings=[emb],
        documents=[text],
        metadatas=[metadata[name]]
    )

# ----------------------------------------------------
# Conflict-Aware Retrieval (Core Logic)
# ----------------------------------------------------
def retrieve(query, role):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    scored = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        score = meta["priority"] if meta["role"] == role else 0
        scored.append((doc, meta, score))

    return sorted(scored, key=lambda x: x[2], reverse=True)

# ----------------------------------------------------
# Ask Function: Gemini → Llama fallback
# ----------------------------------------------------
def ask(query, role="intern"):

    retrieved = retrieve(query, role)

    # Only select top-2 to prevent prompt explosion
    top_docs = retrieved[:2]

    context = ""
    for doc, meta, score in top_docs:
        shortened = doc[:1500]  # safety cutoff
        context += f"[role={meta['role']} priority={meta['priority']}]\n{shortened}\n\n"

    prompt = f"""
You are a conflict-aware HR policy assistant for NebulaGears.

User Role: {role}
Documents (ranked by conflict-resolution):
{context}

Rules:
1. When documents contradict each other, prioritize the one matching the user’s role.
2. If multiple documents match, choose the one with higher priority.
3. Always cite EXACT source document names.

User Query: {query}
"""

    # -------------------- TRY GEMINI FIRST --------------------
    try:
        if GEMINI_KEY:
            model = genai.GenerativeModel("gemini-2.0-flash")
            out = model.generate_content(prompt)
            return "Gemini Response:\n" + out.text
    except Exception as e:
        print("⚠ Gemini failed:", e)
        print("→ Switching to Llama")

    # -------------------- LLAMA FALLBACK --------------------
    if local_llm:
        ans = local_llm(prompt, max_tokens=250)
        return "Llama Response:\n" + ans["choices"][0]["text"]

    return "❌ No model available."

# ----------------------------------------------------
# TEST QUERY (Required in Assignment)
# ----------------------------------------------------
if __name__ == "__main__":
    print(ask("I just joined as a new intern. Can I work from home?", role="intern"))
