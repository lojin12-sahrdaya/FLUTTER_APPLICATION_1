import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Import your document loader that uses DirectoryLoader
from document_loader import load_documents  # Make sure this file is in same folder or PYTHONPATH

# Constants for saving index and data
INDEX_SAVE_PATH = 'qa_index.faiss'
DATA_SAVE_PATH = 'qa_data.json'


def build_faiss_index(texts, model):
    """
    Generate embeddings for all texts and build a FAISS L2 index.
    """
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index


def save_index_and_docs(index, docs, index_path, data_path):
    """
    Save the FAISS index and document JSON data to disk.
    """
    print(f"Saving FAISS index to '{index_path}'...")
    faiss.write_index(index, index_path)

    print(f"Saving document metadata to '{data_path}'...")
    # Extract simple metadata to save
    simple_docs = []
    for doc in docs:
        simple_docs.append({
            "title": os.path.basename(doc.metadata.get("source", "unknown.txt")),
            "content": doc.page_content
        })

    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(simple_docs, f, ensure_ascii=False, indent=2)


def main():
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading documents using document_loader...")
    docs = load_documents()  # Returns list of Documents with page_content and metadata

    texts = [doc.page_content for doc in docs]

    index = build_faiss_index(texts, model)

    save_index_and_docs(index, docs, INDEX_SAVE_PATH, DATA_SAVE_PATH)

    print("Setup complete! Index and docs are ready for your app.")


if __name__ == "__main__":
    main()
