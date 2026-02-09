
# === Imports ===
import os
import math
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

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

    print("\n=== Embedding Inspector Lab ===")
    print("Storing three test sentences in the vector store...\n")

    sentences = [
        # Animals and pets
        "The dog barked loudly in the yard.",
        "Cats love to nap in sunny spots.",
        "Birds chirp early in the morning.",
        "Dogs are common pets and can make a lot of noise.",
        "A puppy wagged its tail happily.",
        # Science and physics
        "Electrons spin around the nucleus in atoms.",
        "Gravity keeps planets in orbit around the sun.",
        "The scientist observed the chemical reaction.",
        # Food and cooking
        "Fresh bread smells wonderful when it is baking.",
        "Tomatoes are used in many Italian recipes.",
        "Cooking pasta requires boiling water.",
        # Sports and activities
        "Soccer players run across the field to score goals.",
        "Swimming is a great way to exercise.",
        "The tennis match lasted for hours.",
        # Weather and nature
        "Rain fell gently on the green leaves.",
        "Thunderstorms can be loud and frightening.",
        # Technology and programming
        "Python is a popular programming language for data science.",
        "The computer crashed during the software update.",
        "Artificial intelligence is changing the world."
    ]

    now = datetime.datetime.now().isoformat()
    categories = [
        "animals", "animals", "animals", "animals", "animals",
        "science", "science", "science",
        "food", "food", "food",
        "sports", "sports", "sports",
        "nature", "nature",
        "technology", "technology", "technology"
    ]
    metadatas = [
        {"created_at": now, "index": idx, "category": categories[idx]}
        for idx in range(len(sentences))
    ]

    vector_store.add_texts(sentences, metadatas=metadatas)

    print(f"✅ Stored {len(sentences)} sentences in the vector store.")
    print("Sentences added:")
    for sentence in sentences:
        print(f"- {sentence}")

    print("\n=== Semantic Search ===")
    while True:
        user_query = input("Enter a search query (or 'quit' to exit): ").strip()
        if user_query.lower() in {"quit", "exit"}:
            print("Goodbye! 👋")
            break
        if not user_query:
            continue
        user_category = input("Enter a category to filter (or leave blank for all): ").strip().lower()
        if not user_category:
            search_sentences.category = None
        else:
            search_sentences.category = user_category
        search_sentences(vector_store, user_query)
        print()

if __name__ == "__main__":
    main()
