---

# **Conflict-Aware RAG System — NebulaGears Assessment**

**Author:** Ayush
**Role:** GenAI Intern Candidate (NTT AT)

---

## **Overview**

Real companies contain conflicting and outdated documents.
Your task: Build a **RAG (Retrieval-Augmented Generation)** system that can:

1. Retrieve relevant documents
2. Detect contradictory policies
3. Resolve conflicts based on **user role**
4. Produce a final answer with **correct ruling + source citation**

This project implements a **Conflict-Aware RAG System** using:

* **LLM:** Google **Gemini Flash 2.5** (free tier via Google AI Studio)
* **Vector DB:** **ChromaDB (Local)**
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Python stack:** ChromaDB + google-generativeai + SentenceTransformers

---

## **Dataset (3 Conflicting Documents)**

| Document | Filename                    | Summary                                       |
| -------- | --------------------------- | --------------------------------------------- |
| A        | `employee_handbook_v1.txt`  | Says *everyone* can work remotely 100%        |
| B        | `manager_updates_2024.txt`  | Restricts remote work to 3 days/week          |
| C        | `intern_onboarding_faq.txt` | **Interns must be in the office 5 days/week** |

---

## **Core Challenge**

Query:

> **“I just joined as a new intern. Can I work from home?”**

Naive RAG retrieves A or B (because of “work”, “remote”, “home”).
Correct answer must come from **Document C**, because role = intern.

This system solves this problem.

---

# **Features**

✔ Role-aware reasoning (“intern”, “manager”, “employee”)
✔ Conflict detection across documents
✔ Conflict resolution (intern > employee > general policy)
✔ ChromaDB local vector store
✔ Embeddings using SentenceTransformers
✔ Google Gemini Flash 2.5 reasoning
✔ Source citation
✔ Clean modular Python code

---

# **Tech Stack**

### **LLM**

* **Google Gemini Flash 2.5**
  Used via:

```python
model="gemini-1.5-flash"
```

### **Vector DB**

*  **ChromaDB (Local Only)**
  Stored via DuckDB + Parquet inside:

```
./chroma_store/
```

### **Embeddings**

* **SentenceTransformers – all-MiniLM-L6-v2**
  Chosen because:
* Fast
* Open-source
* Works offline
* Bonus points for using open-source models

---

# **Project Structure**

```
Conflict-Aware/
│── rag_conflict_aware.py
│── requirements.txt
│── README.md
│── chroma_store/
│── data/
│     ├── employee_handbook_v1.txt
│     ├── manager_updates_2024.txt
│     └── intern_onboarding_faq.txt
```

---

# **Conflict Logic (The Heart of the System)**

### **Step 1 — Retrieve Top-K Documents**

We retrieve 3 documents from Chroma using cosine similarity.
This may include conflicting documents A, B, and C.

### **Step 2 — Role Detection**

Gemini is asked to detect role from query:

Example:

* “I just joined as a new intern” → role = **intern**

### **Step 3 — Conflict Resolution Rules**

Rules (implemented in prompt + reasoning):

1. **Intern-specific rules override all employee/general rules**
2. **Most recent policy overrides older policy**
3. **Most specific policy overrides general policy**

### Example

* Document A → general (outdated)
* Document B → updated employee-specific
* **Document C → intern-specific (highest priority)** ← correct answer

### **Step 4 — Final Answer With Explicit Citation**

Example output:

> **Final ruling: As an intern at NebulaGears, you must work from the office 5 days a week. Remote work is not permitted.**
>
> **Source: intern_onboarding_faq.txt (Document C)**

---

# **How to Run**

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Set Gemini API Key**

```
set GEMINI_API_KEY=your_key_here   # Windows
```

### **3. Run the Script**

```
python rag_conflict_aware.py
```

---

#  **Required Screenshot (Provided Below)**

**Query:**

> "I just joined as a new intern. Can I work from home?"

**Output (Example):**
✔ Shows “Interns cannot work from home.”
✔ Shows **Document C is the source**
✔ Shows reasoning

 **![Terminal Output](ScreenShot.png)**

---

# **Cost Analysis (10,000 Docs + 5,000 Queries/Day)**

### **Gemini Flash 2.5 Pricing**

* Input tokens: ~$0.075 per 1M
* Output tokens: ~$0.30 per 1M
* Embeddings: FREE (text-embedding-004)

### **Estimated Usage**

* Avg query length: 600 tokens
* 5,000 queries/day → **3,000,000 tokens**

### **Cost Calculation**

* 3M tokens ≈ 0.003 * $0.075 = **$0.225/day**
* Monthly: **≈ $6.75**

### **Conclusion**

Running 5,000 queries/day costs **under $7 per month** using Gemini Flash 2.5.

---

# Bonus: Open-Source LLM Support (Optional)

For bonus points, the system also supports:

### **Llama 3.1-8B (GGUF) on Colab**

* Uses `llama.cpp`
* Loads via GGUF quantized file
* Works without GPU

Colab link (example placeholder):

```
https://colab.research.google.com/drive/XXXXX
```

Add to README:

```
python open_source_rag.py
```
