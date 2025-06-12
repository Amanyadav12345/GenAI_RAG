import os
import faiss
import numpy as np
import pickle

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import llama_cpp
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PDF_PATH = "seq2seq.pdf"
EMBEDDINGS_FILE = "seq2seq_texts.pkl"
FAISS_INDEX_FILE = "seq2seq_index.faiss"
GGUF_MODEL_PATH = "mistral-7b-v0.1.Q4_K_M.gguf"


def extract_and_embed():
    print("Extracting and embedding PMBOK content...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    texts = [doc.page_content for doc in chunks]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Save texts
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(texts, f)

    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    print("Embedding and indexing complete.")


def load_index_and_texts():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(EMBEDDINGS_FILE, "rb") as f:
        texts = pickle.load(f)
    return index, texts


def load_llm():
    print("Loading quantized language model...")
    return Llama(model_path=GGUF_MODEL_PATH, n_ctx=2048, n_threads=4)


def query_qa_system(user_input, index, texts, embed_model, llm, top_k=3):
    q_embed = embed_model.encode([user_input])
    D, I = index.search(np.array(q_embed), k=top_k)
    context = "\n\n".join([texts[i] for i in I[0]])

    prompt = f"""You are a PMBOK expert. Answer based only on the context below.

Context:
{context}

Question: {user_input}
Answer:"""

    output = llm(prompt, stop=["\nQuestion:"], max_tokens=256)
    return output["choices"][0]["text"].strip()


def main():
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(EMBEDDINGS_FILE):
        extract_and_embed()

    index, texts = load_index_and_texts()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = load_llm()

    print("\n✅ PMBOK QA System Ready.")
    print("Type your question or 'exit' to quit.\n")

    while True:
        user_input = input("❓ Ask PMBOK: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = query_qa_system(user_input, index, texts, embed_model, llm)
        print(f"\n💬 Answer: {response}\n")


if __name__ == "__main__":
    main()
