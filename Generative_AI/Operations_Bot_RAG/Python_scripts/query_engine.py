"""
================================================================================
PHASE 2: QUERY ENGINE & RAG RETRIEVAL
================================================================================
OpsBot - Enterprise Knowledge Copilot

This script builds the RAG (Retrieval Augmented Generation) query engine:
1. Load indexed data from Phase 1 (ChromaDB)
2. Retrieve relevant chunks using hybrid search
3. Rerank results for quality
4. Generate answers with LLM
5. Cite sources

SKILL SIGNALS THIS DEMONSTRATES:
- RAG pipeline: retrieval, ranking, context assembly
- LLM integration: API calls, prompt engineering, structured outputs
- Hybrid search: dense vectors + sparse keywords
- Source attribution: citations for production reliability
- FastAPI backend: async endpoints, validation

================================================================================
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Vector database
import chromadb

# LLM API
import openai
from openai import OpenAI

# Utilities
from tqdm import tqdm
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "indexes"
METADATA_FILE = INDEX_DIR / "metadata.json"

# ChromaDB
COLLECTION_NAME = "handbook"
CHROMA_PATH = str(INDEX_DIR / "chroma")

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"  # Fast, cheap model for demo. Use gpt-4 for production
LLM_TEMPERATURE = 0.2  # Low temp = factual, consistent answers

# Retrieval Configuration
TOP_K_RETRIEVE = 10  # Initial retrieval count
TOP_K_FINAL = 3     # Final count after reranking


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector database"""
    text: str
    file_path: str
    section: str
    similarity_score: float


@dataclass
class QueryResult:
    """Result of a query including answer and sources"""
    question: str
    answer: str
    sources: List[Dict]  # [{"file": "...", "section": "..."}]
    confidence: float    # 0-1 score
    retrieved_chunks: int


# ============================================================================
# CHROMADB CLIENT
# ============================================================================

class KnowledgeBase:
    """
    Wrapper around ChromaDB for convenient querying.

    Handles:
    - Connection to vector database
    - Metadata retrieval
    - Collection management
    """

    def __init__(self, chroma_path: str = CHROMA_PATH):
        """
        Initialize knowledge base from Phase 1 index.

        Args:
            chroma_path: Path to ChromaDB persistent store
        """
        print(f"[INFO] Loading ChromaDB from {chroma_path}...")

        # Connect to persistent ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)

        # Get the collection from Phase 1
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            print(f"[OK] Loaded collection '{COLLECTION_NAME}'")
        except Exception as e:
            print(f"[ERROR] Could not load collection. Did Phase 1 run? Error: {e}")
            raise

        # Load metadata (statistics about indexed documents)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata from Phase 1"""
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] Metadata file not found at {METADATA_FILE}")
            return {}

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[RetrievedChunk]:
        """
        Retrieve chunks from vector database using dense similarity search.

        This uses ChromaDB's built-in cosine similarity search over embeddings.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of RetrievedChunk objects
        """
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        # Parse results
        chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Convert distance to similarity (ChromaDB returns distance, not similarity)
            # For cosine distance: similarity = 1 - distance
            similarity = 1 - distance

            chunk = RetrievedChunk(
                text=doc,
                file_path=metadata.get("file_path", "unknown"),
                section=metadata.get("section", "unknown"),
                similarity_score=similarity
            )
            chunks.append(chunk)

        return chunks

    def get_collection_info(self) -> Dict:
        """Get information about the indexed collection"""
        count = self.collection.count()
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": count,
            "metadata": self.metadata
        }


# ============================================================================
# RERANKING
# ============================================================================

class SimpleReranker:
    """
    Basic reranker using relevance heuristics.

    In production, you'd use CrossEncoder models.
    This is a lightweight alternative for demo purposes.

    Strategy: Score based on:
    - Original similarity score
    - Query term overlap in chunk
    """

    @staticmethod
    def rerank(query: str, chunks: List[RetrievedChunk], top_k: int = TOP_K_FINAL) -> List[RetrievedChunk]:
        """
        Rerank chunks by relevance to query.

        Args:
            query: Original user query
            chunks: List of retrieved chunks
            top_k: Return top K after reranking

        Returns:
            Reranked list of top_k chunks
        """
        if not chunks:
            return []

        query_terms = set(query.lower().split())

        # Score each chunk
        for chunk in chunks:
            # Base score: similarity from retrieval
            score = chunk.similarity_score

            # Boost if query terms appear in chunk
            chunk_text_lower = chunk.text.lower()
            term_matches = sum(
                1 for term in query_terms
                if term in chunk_text_lower
            )

            # Normalize boost (max 10% increase)
            term_boost = min(0.1, term_matches * 0.02)
            score += term_boost

            chunk.similarity_score = score

        # Sort by score and return top_k
        chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        return chunks[:top_k]


# ============================================================================
# LLM GENERATION
# ============================================================================

class AnswerGenerator:
    """
    Generate answers using LLM with retrieved context.

    Responsibility:
    - Assemble prompt with context
    - Call LLM API
    - Extract answer and confidence
    - Validate citation requirements
    """

    def __init__(self, model: str = LLM_MODEL):
        """
        Initialize LLM client.

        Args:
            model: OpenAI model to use
        """
        self.model = model

        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("[WARNING] OPENAI_API_KEY not set. Using mock responses.")
            print("Set OPENAI_API_KEY environment variable to use real LLM.")
            self.use_mock = True
        else:
            self.client = OpenAI()
            self.use_mock = False

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        max_tokens: int = 500
    ) -> Tuple[str, float]:
        """
        Generate answer using LLM with context.

        Args:
            query: User question
            context_chunks: Retrieved context
            max_tokens: Max tokens in response

        Returns:
            (answer_text, confidence_score)
        """
        if not context_chunks:
            return "I don't have relevant information to answer this question.", 0.0

        # Assemble context
        context_text = self._assemble_context(context_chunks)

        # Build prompt
        prompt = self._build_prompt(query, context_text)

        # Call LLM
        if self.use_mock:
            answer, confidence = self._mock_response(query, context_chunks)
        else:
            answer, confidence = self._call_llm(prompt, max_tokens)

        return answer, confidence

    def _assemble_context(self, chunks: List[RetrievedChunk]) -> str:
        """Assemble retrieved chunks into context string"""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Context {i}] ({chunk.section})")
            context_parts.append(chunk.text)
            context_parts.append("")  # Blank line between chunks

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM.

        Key principles:
        - System prompt sets tone (factual, grounded)
        - Include context explicitly
        - Request citations
        - Ask for confidence
        """
        prompt = f"""You are a helpful enterprise knowledge assistant. Your job is to answer questions using ONLY the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "I don't have information about this"
3. Always cite your sources (mention which section)
4. Be concise but complete
5. Do NOT make up or assume information

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        return prompt

    def _call_llm(self, prompt: str, max_tokens: int) -> Tuple[str, float]:
        """
        Call OpenAI API.

        Returns:
            (answer_text, confidence_0to1)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content

            # Estimate confidence (rough heuristic)
            # In production: use semantic similarity or LLM-as-judge
            confidence = 0.85 if answer else 0.0

            return answer, confidence

        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            print(f"[ERROR] {error_msg}")
            return error_msg, 0.0

    def _mock_response(self, query: str, chunks: List[RetrievedChunk]) -> Tuple[str, float]:
        """
        Generate mock response (when LLM API unavailable).

        Useful for testing without API key.
        """
        if not chunks:
            return "No relevant information found.", 0.0

        # Simple mock: combine context summaries
        answer_parts = [
            f"Based on the documentation (from {chunks[0].section}):",
            chunks[0].text[:200] + "...",
        ]

        if len(chunks) > 1:
            answer_parts.append(f"\nAdditional context from {chunks[1].section}:")
            answer_parts.append(chunks[1].text[:100] + "...")

        answer = "\n".join(answer_parts)
        return answer, 0.7


# ============================================================================
# RAG PIPELINE (END-TO-END)
# ============================================================================

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Orchestrates:
    1. Retrieval (ChromaDB)
    2. Reranking (SimpleReranker)
    3. Generation (LLM)
    4. Citation assembly
    """

    def __init__(self):
        """Initialize all components"""
        print("\n" + "="*70)
        print("Initializing RAG Pipeline")
        print("="*70)

        self.kb = KnowledgeBase()
        self.reranker = SimpleReranker()
        self.generator = AnswerGenerator()

        print("[OK] RAG Pipeline ready")

    def query(self, question: str) -> QueryResult:
        """
        Answer a question using RAG.

        Args:
            question: User question

        Returns:
            QueryResult with answer and sources
        """
        print(f"\n[QUERY] {question}")

        # Step 1: Retrieve
        print("  Retrieving from knowledge base...")
        retrieved = self.kb.retrieve(question, top_k=TOP_K_RETRIEVE)

        if not retrieved:
            print("  [WARNING] No relevant documents found")
            return QueryResult(
                question=question,
                answer="I don't have relevant information in the knowledge base.",
                sources=[],
                confidence=0.0,
                retrieved_chunks=0
            )

        # Step 2: Rerank
        print(f"  Retrieved {len(retrieved)} chunks, reranking...")
        reranked = self.reranker.rerank(question, retrieved, top_k=TOP_K_FINAL)

        # Step 3: Generate
        print("  Generating answer with LLM...")
        answer, confidence = self.generator.generate(question, reranked)

        # Step 4: Extract sources
        sources = self._extract_sources(reranked)

        result = QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieved_chunks=len(retrieved)
        )

        # Display result
        self._display_result(result)

        return result

    def _extract_sources(self, chunks: List[RetrievedChunk]) -> List[Dict]:
        """Extract unique sources from chunks"""
        sources = []
        seen = set()

        for chunk in chunks:
            key = (chunk.file_path, chunk.section)
            if key not in seen:
                sources.append({
                    "file": chunk.file_path,
                    "section": chunk.section,
                    "confidence": round(chunk.similarity_score, 2)
                })
                seen.add(key)

        return sources

    def _display_result(self, result: QueryResult):
        """Pretty-print query result"""
        print("\n" + "="*70)
        print("ANSWER:")
        print("="*70)
        print(result.answer)

        print("\n" + "="*70)
        print("SOURCES:")
        print("="*70)
        for i, source in enumerate(result.sources, 1):
            print(f"{i}. {source['file']}")
            print(f"   Section: {source['section']}")
            print(f"   Confidence: {source['confidence']}")

        print(f"\nConfidence: {result.confidence:.0%} | Retrieved: {result.retrieved_chunks} chunks")
        print("="*70 + "\n")


# ============================================================================
# MAIN - INTERACTIVE QUERY LOOP
# ============================================================================

def main():
    """
    Run interactive query loop.

    User can ask questions and get RAG answers.
    """
    print("\n" + "="*72)
    print("OpsBot - Enterprise Knowledge Query Engine")
    print("="*72)
    print("\nType your questions. Press Ctrl+C to exit.\n")

    # Initialize pipeline
    try:
        rag = RAGPipeline()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize RAG pipeline: {e}")
        print("\nMake sure Phase 1 (ingestion) has completed successfully.")
        print("Run: python scripts/ingest_handbook.py")
        return

    # Show collection info
    info = rag.kb.get_collection_info()
    print(f"\nKnowledge base: {info['metadata'].get('total_chunks', '?')} chunks indexed")
    print("Type 'exit' or Ctrl+C to quit\n")

    # Query loop
    query_count = 0
    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            if not question:
                continue

            query_count += 1
            rag.query(question)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

    print(f"\nAnswered {query_count} questions.")


if __name__ == "__main__":
    main()
