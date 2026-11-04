import os
import chromadb

# Absolute paths to ensure clean execution
KB_DIR = r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\01_Knowledge_Base\cleaned5"
DB_DIR = r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\02_RAG_Pipeline\chroma_db"

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Slices a document into overlapping text blocks.
    - chunk_size: 800 characters (~150-200 words)
    - overlap: 150 characters (preserves sentence continuity across chunks)
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

def build_vector_store():
    # 1. Connect to local ChromaDB (persisted on disk)
    client = chromadb.PersistentClient(path=DB_DIR)

    # Recreate the collection so re-running this script produces a fresh index
    collection_name = "langchain_kb"
    try:
        client.delete_collection(name=collection_name)
        print(f"Cleared existing collection: '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    # 2. Read the 10 cleaned Markdown files
    if not os.path.exists(KB_DIR):
        raise FileNotFoundError(f"Directory not found: {KB_DIR}")

    files = sorted([f for f in os.listdir(KB_DIR) if f.endswith(".md")])
    print(f"Found {len(files)} cleaned documents to index.\n")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for filename in files:
        filepath = os.path.join(KB_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        file_chunks = chunk_text(content, chunk_size=800, overlap=150)
        print(f" - {filename}: {len(file_chunks)} chunks")

        for idx, chunk in enumerate(file_chunks):
            # Unique ID for every chunk (e.g., 01_models.md_chunk_0)
            chunk_id = f"{filename}_chunk_{idx}"
            
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename, "chunk_index": idx})
            all_ids.append(chunk_id)

    # 3. Add to ChromaDB
    # ChromaDB uses its built-in embedding model (all-MiniLM-L6-v2) by default
    print(f"\nEmbedding and storing {len(all_chunks)} chunks in ChromaDB...")
    collection.add(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print(f"\nIndex successfully built and saved to: {DB_DIR}")

if __name__ == "__main__":
    build_vector_store()