"""
================================================================================
PHASE 1: DOCUMENT INGESTION & INDEXING
================================================================================
OpsBot - Enterprise Knowledge Copilot

This script handles the end-to-end ingestion pipeline:
1. Clone the GitLab handbook from the public repository
2. Parse all Markdown files
3. Chunk documents hierarchically (by heading structure)
4. Generate embeddings for semantic search
5. Store everything in ChromaDB (vector database)

After running this script, you'll have:
- A local copy of the handbook in data/handbook/
- An indexed ChromaDB collection in data/indexes/
- Metadata about all documents (for citations and filtering)

SKILL SIGNALS THIS DEMONSTRATES:
- Data engineering: parsing, chunking, ETL pipeline
- RAG fundamentals: embedding strategy, vector store setup
- Production mindset: chunking by semantic boundaries, metadata preservation

================================================================================
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import re

# For embeddings - using open-source transformers (free, local)
# We'll use a lightweight approach with transformers library
from transformers import AutoTokenizer, AutoModel
import torch

# For vector database
import chromadb


# ============================================================================
# CONFIGURATION
# ============================================================================

# Root directory for the project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HANDBOOK_DIR = DATA_DIR / "handbook"
INDEX_DIR = DATA_DIR / "indexes"
METADATA_FILE = INDEX_DIR / "metadata.json"

# GitHub repository for GitLab's public handbook
# Updated: The correct repo is the gitlab-com/content-sites-handbook
# Using the correct public repository URL
HANDBOOK_REPO = "https://gitlab.com/gitlab-com/content-sites-handbook.git"

# Embedding model - small, fast, free, runs locally
# Uses HuggingFace's sentence transformers
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB collection name
COLLECTION_NAME = "handbook"

# ============================================================================
# STEP 1: CLONE/UPDATE HANDBOOK REPOSITORY
# ============================================================================

def clone_or_update_handbook():
    """
    Clone the GitLab handbook repository if it doesn't exist.
    If it exists, pull the latest version.

    The handbook is completely public at:
    https://github.com/gitlab-com/content-sites-handbook

    This gives us real enterprise content: policies, SOPs, escalation procedures, etc.
    """
    print("\n" + "="*70)
    print("STEP 1: Cloning/updating GitLab handbook repository...")
    print("="*70)

    HANDBOOK_DIR.mkdir(parents=True, exist_ok=True)

    # Check if handbook directory has files
    markdown_files = list(HANDBOOK_DIR.glob("**/*.md"))

    if markdown_files:
        # Handbook already exists with content
        print(f"[OK] Handbook exists at {HANDBOOK_DIR}")
        print(f"  Found {len(markdown_files)} markdown files")
    elif (HANDBOOK_DIR / ".git").exists():
        # Git repository exists - update it
        print(f"[OK] Handbook git repo exists at {HANDBOOK_DIR}")
        print("  Pulling latest changes...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=HANDBOOK_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[OK] Handbook updated successfully")
        else:
            print(f"[WARNING] Git pull encountered an issue (this is okay):\n{result.stderr}")
    else:
        # Try to clone
        print(f"Cloning from {HANDBOOK_REPO}")
        print("This may take 1-2 minutes on first run...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", HANDBOOK_REPO, str(HANDBOOK_DIR)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[OK] Handbook cloned successfully")
        else:
            print(f"[WARNING] Could not clone from remote (using local files if available)")
            # Continue anyway - might have local handbook files


# ============================================================================
# STEP 2: PARSE MARKDOWN FILES
# ============================================================================

def find_markdown_files() -> List[Path]:
    """
    Recursively find all .md files in the handbook directory.

    Returns:
        List of Path objects pointing to all markdown files
    """
    print("\n" + "="*70)
    print("STEP 2: Finding Markdown files in handbook...")
    print("="*70)

    markdown_files = list(HANDBOOK_DIR.rglob("*.md"))

    # Filter out non-content files
    # (e.g., README.md in root, .github files, etc.)
    markdown_files = [
        f for f in markdown_files
        if ".github" not in str(f) and "__pycache__" not in str(f)
    ]

    print(f"[OK] Found {len(markdown_files)} markdown files")
    return sorted(markdown_files)


def parse_markdown_file(file_path: Path) -> Dict:
    """
    Parse a single Markdown file and extract its content and metadata.

    Args:
        file_path: Path to the .md file

    Returns:
        Dictionary with:
        - file_path: original file path (for citations)
        - title: file name
        - content: full file content
        - relative_path: path relative to handbook root (for organization)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Get relative path for better organization (e.g., "IT/onboarding.md")
        relative_path = file_path.relative_to(HANDBOOK_DIR)

        return {
            "file_path": str(file_path),
            "relative_path": str(relative_path),
            "title": file_path.stem,
            "content": content
        }
    except Exception as e:
        print(f"[WARNING] Skipping {file_path}: {e}")
        return None


# ============================================================================
# STEP 3: INTELLIGENT CHUNKING STRATEGY
# ============================================================================

def chunk_markdown_hierarchically(file_data: Dict) -> List[Dict]:
    """
    Split a markdown file into chunks while preserving semantic structure.

    Strategy: Chunk by heading hierarchy (H1 → H2 → H3)
    - This preserves context better than random splitting
    - Financial docs have structure (Risk Factors, MD&A, etc.)
    - Maintains semantic boundaries

    For example, in IT policies:
    ## Security Policies        ← Chunk boundary
    ### Password Requirements

    This ensures a RAG query about passwords retrieves the full security policy context.

    Args:
        file_data: Dictionary with file content and metadata

    Returns:
        List of chunk dictionaries, each with:
        - text: the chunk content
        - file_path, relative_path, section: metadata for citations
        - chunk_id: unique identifier
    """
    content = file_data["content"]
    file_path = file_data["file_path"]
    relative_path = file_data["relative_path"]

    chunks = []
    chunk_id = 0

    # Split by h2 headings (##) - these are major section boundaries
    # Pattern: ## Section Name followed by content until next ##
    h2_pattern = r"(^## .+?)(?=^## |\Z)"
    sections = re.split(h2_pattern, content, flags=re.MULTILINE)

    current_h2 = "Introduction"  # Default if no h2 found

    for i, section in enumerate(sections):
        section = section.strip()

        if not section:
            continue

        # Check if this is a heading (lines ending with pattern are headings)
        if section.startswith("## "):
            # This is an h2 heading
            current_h2 = section.replace("## ", "").strip()
            continue

        # Skip if section is too short (noise)
        if len(section) < 50:
            continue

        # Further split by h3 headings if they exist
        h3_pattern = r"(^### .+?)(?=^### |^## |\Z)"
        subsections = re.split(h3_pattern, section, flags=re.MULTILINE)

        current_h3 = None
        for j, subsection in enumerate(subsections):
            subsection = subsection.strip()

            if not subsection:
                continue

            if subsection.startswith("### "):
                current_h3 = subsection.replace("### ", "").strip()
                continue

            if len(subsection) < 50:
                continue

            # Create chunk with metadata
            section_label = f"{current_h2}"
            if current_h3:
                section_label += f" → {current_h3}"

            chunk = {
                "text": subsection,
                "file_path": file_path,
                "relative_path": relative_path,
                "section": section_label,
                "chunk_id": f"{relative_path}#{chunk_id}",
                "token_estimate": len(subsection.split())  # Rough token count
            }

            chunks.append(chunk)
            chunk_id += 1

    # If no h2/h3 chunks were created, create one chunk for the whole file
    if not chunks and len(content) > 100:
        chunks.append({
            "text": content,
            "file_path": file_path,
            "relative_path": relative_path,
            "section": file_data["title"],
            "chunk_id": f"{relative_path}#0",
            "token_estimate": len(content.split())
        })

    return chunks


# ============================================================================
# STEP 4: GENERATE EMBEDDINGS
# ============================================================================

def load_embedding_model(model_name: str):
    """
    Load the embedding model for generating embeddings.

    Model: sentence-transformers/all-MiniLM-L6-v2 (via transformers)
    - 384 dimensions
    - 22MB (fits easily in memory)
    - Fast inference (cpu-friendly)
    - Good semantic understanding for enterprise docs
    - Completely free and open-source

    First run will download ~50MB from HuggingFace (only once, then cached locally)

    Args:
        model_name: Name of the model on HuggingFace Hub

    Returns:
        Loaded model wrapper with encode method
    """
    print(f"\n[OK] Loading embedding model: {model_name}")
    print("  (First run downloads ~50MB from HuggingFace - only once)")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode (no gradients needed)

    # Move to CPU (ensure no CUDA errors if GPU not available)
    device = torch.device("cpu")
    model.to(device)

    class EmbeddingWrapper:
        """Wrapper to provide sentence-transformers-like interface"""
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

        def encode(self, texts):
            """
            Encode texts to embeddings

            Args:
                texts: List of strings or single string

            Returns:
                List of embedding vectors (lists)
            """
            if isinstance(texts, str):
                texts = [texts]

            # Tokenize
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy().tolist()

    wrapper = EmbeddingWrapper(model, tokenizer, device)
    print(f"[OK] Model loaded. Embedding dimension: 384")
    return wrapper


# ============================================================================
# STEP 5: INITIALIZE CHROMADB
# ============================================================================

def initialize_chromadb():
    """
    Initialize ChromaDB vector database.

    ChromaDB:
    - Runs in-process (no server needed)
    - Persists to disk
    - Supports metadata filtering
    - Built for RAG applications

    Returns:
        (client, collection) tuple
    """
    print("\n" + "="*70)
    print("STEP 5: Initializing ChromaDB vector store...")
    print("="*70)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Use new ChromaDB API (v0.4+)
    # Store data in data/indexes/chroma/
    chroma_path = str(INDEX_DIR / "chroma")

    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection if it exists (for fresh starts)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet

    # Create new collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    print(f"[OK] ChromaDB initialized with collection '{COLLECTION_NAME}'")
    return client, collection


# ============================================================================
# STEP 6: INGEST ALL DOCUMENTS
# ============================================================================

def ingest_documents(collection, embedding_model):
    """
    Main ingestion loop:
    1. Find all markdown files
    2. Parse each file
    3. Chunk hierarchically
    4. Generate embeddings
    5. Store in ChromaDB

    Args:
        collection: ChromaDB collection object
        embedding_model: Loaded SentenceTransformer model

    Returns:
        Metadata dictionary for later reference
    """
    print("\n" + "="*70)
    print("STEP 3-6: Parsing, chunking, and indexing documents...")
    print("="*70)

    markdown_files = find_markdown_files()

    all_chunks = []
    metadata_records = []

    # Process each file
    for file_path in tqdm(markdown_files, desc="Processing files"):
        file_data = parse_markdown_file(file_path)
        if not file_data:
            continue

        # Chunk the file hierarchically
        chunks = chunk_markdown_hierarchically(file_data)
        all_chunks.extend(chunks)

        # Track metadata for this file
        metadata_records.append({
            "file_path": file_data["file_path"],
            "relative_path": file_data["relative_path"],
            "num_chunks": len(chunks),
            "total_tokens": sum(c["token_estimate"] for c in chunks)
        })

    print(f"\n[OK] Created {len(all_chunks)} chunks from {len(markdown_files)} files")

    # Generate embeddings and store in ChromaDB
    print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
    print("(This takes a few minutes - embeddings are computed locally)")

    batch_size = 50  # Process in batches for memory efficiency

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing chunks"):
        batch = all_chunks[i:i+batch_size]

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]

        # Generate embeddings (local, no API call needed)
        embeddings_result = embedding_model.encode(texts)

        # Handle both numpy arrays and lists
        if hasattr(embeddings_result, 'tolist'):
            embeddings = embeddings_result.tolist()
        else:
            embeddings = embeddings_result  # Already a list

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in batch]
        documents = texts
        metadatas = [
            {
                "file_path": chunk["file_path"],
                "relative_path": chunk["relative_path"],
                "section": chunk["section"],
                "tokens": chunk["token_estimate"]
            }
            for chunk in batch
        ]

        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"\n[OK] Successfully indexed {len(all_chunks)} chunks")

    return {
        "total_files": len(markdown_files),
        "total_chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": 384,
        "collection_name": COLLECTION_NAME,
        "files": metadata_records
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute the complete ingestion pipeline.

    RESUMABLE: If this script fails midway, you can safely re-run it.
    It will:
    - Skip re-cloning if handbook already exists
    - Rebuild ChromaDB from existing files
    - Be idempotent
    """

    print("\n")
    print("="*72)
    print(" "*18 + "OPSBOT PHASE 1: DOCUMENT INGESTION")
    print(" "*16 + "Enterprise Knowledge Copilot - Indexing")
    print("="*72 + "\n")

    try:
        # Step 1: Clone or update handbook
        clone_or_update_handbook()

        # Step 2-3: Parse and chunk documents
        embedding_model = load_embedding_model(EMBEDDING_MODEL)

        # Step 4: Initialize vector database
        client, collection = initialize_chromadb()

        # Step 5-6: Ingest and index
        metadata = ingest_documents(collection, embedding_model)

        # Save metadata for later reference
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        # Summary
        print("\n" + "="*70)
        print("[OK] PHASE 1 COMPLETE!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Files indexed:        {metadata['total_files']}")
        print(f"  Chunks created:       {metadata['total_chunks']}")
        print(f"  Embedding model:      {metadata['embedding_model']}")
        print(f"  Vector dimension:     {metadata['embedding_dimension']}")
        print(f"  ChromaDB location:    {INDEX_DIR / 'chroma'}")
        print(f"  Metadata saved to:    {METADATA_FILE}")
        print("\n" + "="*70)
        print("NEXT: Run Phase 2 to build the RAG query engine")
        print("      $ python scripts/query_engine.py")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Error during ingestion: {e}")
        print("[WARNING] If this fails due to network issues, try again.")
        print("   Your partially indexed data is saved and will resume.")
        raise


if __name__ == "__main__":
    main()
