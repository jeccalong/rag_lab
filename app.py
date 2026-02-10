# === Imports ===
import os
import math
import datetime

from dotenv import load_dotenv

# === DRY RUN MODE ===
DRY_RUN = False  # Set to True to disable API calls and simulate behavior

from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
def load_document(vector_store, file_path: str):
    """
    Loads a text file, creates a LangChain Document, and adds it to the vector store.
    Args:
        vector_store (InMemoryVectorStore): The vector store instance.
        file_path (str): Path to the text file to load.
    Returns:
        str: The document ID if added successfully, else None.
    """
    import os
    from datetime import datetime
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        metadata = {
            "fileName": os.path.basename(file_path),
            "createdAt": datetime.now().isoformat(),
        }
        token_count = count_tokens(text)
        print(f"File '{metadata['fileName']}' has {token_count} tokens and {len(text)} chars.")
        if DRY_RUN:
            print(f"[DRY RUN] Would add '{metadata['fileName']}' to vector store. No API call made.")
            return None
        document = Document(page_content=text, metadata=metadata)
        doc_ids = vector_store.add_documents([document])
        print(f"✅ Loaded '{metadata['fileName']}' ({len(text)} chars) into vector store.")
        return doc_ids[0] if doc_ids else None
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        error_msg = str(e)
        if (
            "maximum context length" in error_msg.lower()
            or "token" in error_msg.lower()
        ):
            print("⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"❌ Error loading document '{file_path}': {error_msg}")
        return None

# === Constants ===
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BASE_URL = "https://models.inference.ai.azure.com"

# === Function Definitions ===
def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    Args:
        vector_a (list[float]): First vector.
        vector_b (list[float]): Second vector.
    Returns:
        float: Cosine similarity between the two vectors.
    Raises:
        ValueError: If vectors are not the same length.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    return dot_product / (norm_a * norm_b)

def search_sentences(vector_store, query: str, k: int = 3):
    """
    Search the vector store for sentences similar to the query.
    Args:
        vector_store (InMemoryVectorStore): The vector store instance.
        query (str): The search query string.
        k (int, optional): Number of results to return. Defaults to 3.
    Returns:
        list: List of (Document, score) tuples for the top k results.
    """
    filter_func = None
    category = getattr(search_sentences, "category", None)
    if category:
        def filter_func(doc):
            return doc.metadata.get("category", "").lower() == category.lower()

    # Vector similarity search
    vector_results = vector_store.similarity_search_with_score(query, k=k, filter=filter_func)

    # Keyword matching
    keyword_results = []
    query_keywords = set(query.lower().split())
    for doc, score in vector_results:
        doc_words = set(doc.page_content.lower().split())
        keyword_match = len(query_keywords & doc_words) > 0
        keyword_results.append((doc, score, keyword_match))

    # Hybrid ranking: prioritize docs with keyword match, then by vector score
    hybrid_results = sorted(keyword_results, key=lambda x: (-x[2], -x[1]))[:k]

    print(f"\nTop {k} hybrid search results for: '{query}'" + (f" in category '{category}'" if category else "") + "\n")
    for rank, (doc, score, keyword_match) in enumerate(hybrid_results, 1):
        match_note = "[Keyword Match]" if keyword_match else ""
        print(f"{rank}. [Score: {score:.4f}] {doc.page_content} {match_note}")
    return [(doc, score) for doc, score, _ in hybrid_results]

# === Token Counting ===
def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """
    Count the number of tokens in a string for a given model.
    Returns 0 if tiktoken is not installed.
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        print("⚠️  tiktoken not installed. Token count will be 0.")
        return 0
    except Exception as e:
        print(f"⚠️  Error counting tokens: {e}")
        return 0

def main():
    """
    Main entry point for the script. Loads environment, checks config, stores and searches embeddings.
    """
    load_dotenv()
    print("🤖 Python LangChain Agent Starting...\n")

    github_token = os.getenv(GITHUB_TOKEN_ENV)
    if not github_token:
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("Please create a .env file with your GitHub token:")
        print("GITHUB_TOKEN=your-github-token-here")
        print("\nGet your token from: https://github.com/settings/tokens")
        print("Or use GitHub Models: https://github.com/marketplace/models")
        return

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL,
        api_key=github_token,
        check_embedding_ctx_length=False,
    )
    print("✅ OpenAIEmbeddings instance created.")

    vector_store = InMemoryVectorStore(embeddings)
    print("✅ InMemoryVectorStore instance created.")

    print("\n=== Loading Documents into Vector Database ===")
    doc_path1 = "HealthInsuranceBrochure.md"
    doc_id1 = load_document(vector_store, doc_path1)
    if doc_id1:
        print(f"Document '{doc_path1}' loaded successfully with ID: {doc_id1}")
    else:
        print(f"Failed to load document: {doc_path1}")

    doc_path2 = "EmployeeHandbook.md"
    doc_id2 = load_document(vector_store, doc_path2)
    if doc_id2:
        print(f"Document '{doc_path2}' loaded successfully with ID: {doc_id2}")
    else:
        print(f"Failed to load document: {doc_path2}")

if __name__ == "__main__":
    main()
