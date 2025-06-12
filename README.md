# 🔍 Mistral-7B RAG System with FAISS + Hugging Face + LangChain

This project demonstrates a **Retrieval-Augmented Generation (RAG)** setup using the powerful **Mistral-7B language model**, with document retrieval powered by **FAISS**, embeddings via **Hugging Face**, and orchestration handled by **LangChain**. It follows best practices in **Generative AI**, focusing on modularity, reproducibility, and secure architecture.

---

## 📦 Features

- 💬 Ask questions about your PDF or document content
- 🔍 Fast semantic search using FAISS
- 🧠 Custom embedding generation with Hugging Face models
- 🧾 Support for large documents using recursive text splitting
- 🛠️ Modular pipeline: ingest → embed → index → generate

---

## 🧰 Tech Stack

| Component        | Tool/Library                  |
|------------------|-------------------------------|
| Language Model   | `Mistral-7B` (GGUF or HuggingFace format) |
| Embeddings       | `sentence-transformers` (Hugging Face) |
| Vector Store     | `FAISS` (Facebook AI Similarity Search) |
| Orchestration    | `LangChain`                   |
| Document Loader  | `PyPDFLoader`                 |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/yourname/mistral-rag
cd mistral-rag
