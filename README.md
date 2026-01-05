# RAG Policy Assistant

An intelligent Question-Answering system designed to navigate complex company policies (Refunds, Shipping, Privacy). This project demonstrates a production-ready **Retrieval-Augmented Generation (RAG)** pipeline featuring **Two-Stage Retrieval (Reranking)** and **Chain-of-Thought (CoT)** reasoning.

---

## Key Features
* **Two-Stage Retrieval:** Uses `ChromaDB` for initial retrieval (k=10) followed by a `Cross-Encoder` (MS-MARCO) to rerank and select the top 5 most relevant chunks.
* **Chain-of-Thought Reasoning:** Custom prompts guide the LLM to "Analyze" -> "Reason" -> "Answer", drastically reducing hallucinations.
* **Self-Correction:** Includes robust error handling and retry logic for API rate limits.
* **Automated Evaluation:** Includes a "LLM-as-a-Judge" suite that grades answers against a ground-truth dataset.

---

## Architecture


1.  **Ingestion:** Documents are loaded and split into 500-character chunks (with overlap) to preserve context.
2.  **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` converts text to vectors.
3.  **Vector Store:** `ChromaDB` stores embeddings for fast similarity search.
4.  **Retrieval Engine:**
    * **Step 1:** Fetch top 10 documents based on cosine similarity.
    * **Step 2:** Rerank using `cross-encoder/ms-marco-MiniLM-L-6-v2` to filter noise.
5.  **Generation:** `gemini-flash-latest` generates the final answer using a strict CoT prompt.

---

## Setup Instructions

**Prerequisites:** Python 3.10+

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/csmishra952/rag-policy-assistant.git](https://github.com/csmishra952/rag-policy-assistant.git)
    cd rag-policy-assistant
    ```

2.  **Install Dependencies**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\Activate
    # Mac/Linux
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_api_key
    ```

4.  **Ingest Data** (Builds the Vector DB)
    ```bash
    python src/ingestion.py
    ```

5.  **Run the Assistant**
    ```bash
    python src/rag_engine.py
    ```

---

## Evaluation Results

The system was evaluated using an automated "LLM-as-a-Judge" pipeline on a diverse test set (Simple Fact, Synthesis, Negative Constraints, and Out-of-Scope).

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Average Accuracy** | **5.0 / 5.0** | Perfect score achieved after implementing Reranking. |
| **Hallucination Rate** | **0%** | Correctly refused to answer "CEO" questions. |
| **Retrieval Quality** | **High** | Successfully retrieved "Stolen Package" policy despite wording differences. |

**Evaluation Script:** `src/evaluate.py`

---

## Prompt Engineering
I evolved the prompt from a simple instruction to a structured **Chain-of-Thought** format to handle complex queries like shipping calculations.

**Prompt Structure:**
```text
You are a Senior Customer Support AI.
<Instructions>
1. **Analyze:** Scan context for keywords (e.g., "stolen" -> "theft").
2. **Reason:** Connect facts (e.g., Canada + $40 -> Flat Rate Shipping).
3. **Answer:** Provide a direct, polite answer.
4. **Refusal:** If unknown, state "I cannot answer based on current policies."
</Instructions>
