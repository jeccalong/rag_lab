# API Efficiency Improvements: Rationale and Outcome
#
# In this version, document chunks are uploaded to the vector store in a single batch API call, rather than one call per chunk.
#
# **Why?**
# - Batch uploading reduces the number of API calls, helping you stay within strict rate limits (15/minute, 300/day).
# - This is especially important for large documents split into many chunks.
#
# **How does this affect output?**
# - You no longer see per-chunk progress messages like "Processing chunk 1/15..." for each chunk.
# - Instead, you see a summary after the batch upload: the total number of chunks and average chunk size.
# - The code is more efficient and less likely to hit API limits, but provides less granular progress feedback.
#
# **If you want per-chunk progress messages:**
# - You can print progress for each chunk before the batch upload, but keep the batch API call for efficiency.
# - This gives user feedback without increasing API usage.
#
# **Summary:**
# - Batch uploads = fewer API calls, more efficient, safer for rate limits.
# - Output is different because the process is optimized for API conservation.
# 
# ---

# === API Call Tracking ===
API_CALL_COUNT = 0

# === Imports ===
import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
# === Constants ===
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BASE_URL = "https://models.inference.ai.azure.com"

# === DRY RUN MODE ===
DRY_RUN = False  # Set to True to disable API calls and simulate behavior

# === Constants ===
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BASE_URL = "https://models.inference.ai.azure.com"

# === DRY RUN MODE ===
DRY_RUN = False  # Set to True to disable API calls and simulate behavior

# === Function Definitions ===

def create_search_tool(vector_store):
    """
    Create a LangChain tool for searching the document repository.
    Args:
        vector_store: The vector store instance.
    Returns:
        Tool: A LangChain tool for document search.
    """
    @tool
    def search_documents(query: str) -> str:
        """
        Searches the company document repository for relevant information based on the given query.
        Use this to find information about company policies, benefits, and procedures.
        """
        results = vector_store.similarity_search_with_score(query, k=3)
        formatted = []
        for idx, (doc, score) in enumerate(results, 1):
            formatted.append(f"Result {idx} (Score: {score:.4f}): {doc.page_content}")
        return "\n\n".join(formatted)
    return search_documents

def load_with_markdown_structure_chunking(vector_store, file_path: str) -> int:
    """
    Split a markdown document by structure (headers), then chunk with overlap.

    Args:
        vector_store: The vector store instance.
        file_path: Path to the markdown document to chunk.

    Returns:
        int: Number of chunks stored.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        )
        header_docs = header_splitter.split_text(text)
        chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        chunks = chunk_splitter.create_documents([doc.page_content for doc in header_docs])
        num_chunks = len(chunks)
        print(f"📄 Loading '{file_path}' with markdown structure chunking...")
        total_stored = load_document_with_chunks(vector_store, file_path, chunks)
        print(f"Total chunks created: {num_chunks}")
        return total_stored
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as error:
        print(f"❌ Error in markdown structure chunking: {error}")
        return 0
    
def load_with_paragraph_chunking(vector_store, file_path: str) -> int:
    """
    Split a document by paragraphs and store chunks in the vector store.

    Args:
        vector_store: The vector store instance.
        file_path: Path to the document to chunk.

    Returns:
        int: Number of chunks stored.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.create_documents([text])
        num_chunks = len(chunks)
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        min_chunk = min(chunk_sizes) if chunk_sizes else 0
        max_chunk = max(chunk_sizes) if chunk_sizes else 0
        newline_starts = sum(1 for chunk in chunks if chunk.page_content.startswith("\n"))
        print(f"📄 Loading '{file_path}' with paragraph-based chunking...")
        total_stored = load_document_with_chunks(vector_store, file_path, chunks)
        print(f"Total chunks created: {num_chunks}")
        print(f"Smallest chunk size: {min_chunk} characters")
        print(f"Largest chunk size: {max_chunk} characters")
        print(f"Chunks starting with newline: {newline_starts}")
        return total_stored
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as error:
        print(f"❌ Error in paragraph chunking: {error}")
        return 0
    
def load_document_with_chunks(vector_store, file_path, chunks):
    """
    Loads chunked documents into the vector store, updating metadata and reporting progress.
    Args:
        vector_store: The vector store instance.
        file_path: Path to the original file.
        chunks: List of LangChain Document objects.
    Returns:
        int: Total number of chunks stored.
    """
    try:
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks, 1):
            chunk.metadata["fileName"] = f"{os.path.basename(file_path)} (Chunk {idx}/{total_chunks})"
            chunk.metadata["createdAt"] = datetime.now().isoformat()
            chunk.metadata["chunkIndex"] = idx
        global API_CALL_COUNT
        if DRY_RUN:
            print(f"[DRY RUN] Would add {total_chunks} chunks to vector store.")
        else:
            vector_store.add_documents(chunks)
            API_CALL_COUNT += 1
            print(f"✅ Successfully loaded {total_chunks} chunks from {os.path.basename(file_path)}")
        return total_chunks
    except Exception as e:
        print(f"❌ Error loading chunks: {e}")
        return 0

def load_document(vector_store, file_path: str):
    """
    Loads a text file, creates a LangChain Document, and adds it to the vector store.
    Args:
        vector_store (InMemoryVectorStore): The vector store instance.
        file_path (str): Path to the text file to load.
    Returns:
        str: The document ID if added successfully, else None.
    """
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
        global API_CALL_COUNT
        document = Document(page_content=text, metadata=metadata)
        doc_ids = vector_store.add_documents([document])
        API_CALL_COUNT += 1
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

def load_with_fixed_size_chunking(vector_store, file_path: str) -> int:
    """
    Split a document into fixed-size chunks and store them in the vector store.

    Args:
        vector_store: The vector store instance.
        file_path: Path to the document to chunk.

    Returns:
        int: Number of chunks stored.

    Raises:
        Exception: If file reading or chunking fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separator=" "
        )
        chunks = splitter.create_documents([text])
        num_chunks = len(chunks)
        avg_chunk_size = (
            sum(len(chunk.page_content) for chunk in chunks) / num_chunks if num_chunks else 0
        )
        print(f"📄 Loading '{file_path}' with fixed-size chunking...")
        total_stored = load_document_with_chunks(vector_store, file_path, chunks)
        print(f"Number of chunks created: {num_chunks}")
        print(f"Average chunk size: {int(avg_chunk_size)} characters")
        return total_stored
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as error:
        print(f"❌ Error in fixed-size chunking: {error}")
        return 0


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

def message_content_to_text(content) -> str:
    """
    Normalize LangChain message content to plain text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)

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

    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url=EMBEDDING_BASE_URL,
        api_key=github_token,
    )
    print("✅ ChatOpenAI instance created.")

    # Create the search tool
    search_tool = create_search_tool(vector_store)

    # Create the agent using LangChain's current API
    agent_executor = create_agent(
        model=chat_model,
        tools=[search_tool],
        system_prompt=(
            "You are a helpful assistant that answers questions about company "
            "policies, benefits, and procedures. Use the search_documents tool "
            "to find relevant information before answering. Always cite which "
            "document chunks you used in your answer."
        ),
    )

    print("\n=== Loading Documents into Vector Database ===")
    doc_path1 = "HealthInsuranceBrochure.md"
    doc_id1 = load_document(vector_store, doc_path1)
    if doc_id1:
        print(f"Document '{doc_path1}' loaded successfully with ID: {doc_id1}")
    else:
        print(f"Failed to load document: {doc_path1}")

    doc_path2 = "EmployeeHandbook.md"
    load_with_markdown_structure_chunking(vector_store, doc_path2)

    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = []
    print("\n=== Agent Chat Interface ===")
    print("You are now chatting with an agent that can answer questions about company policies, benefits, and procedures. The agent will automatically search documents and cite sources.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye! 👋")
            break
        if not user_input:
            continue
        result = agent_executor.invoke(
            {"messages": chat_history + [HumanMessage(content=user_input)]}
        )
        messages = result.get("messages", [])
        response = message_content_to_text(messages[-1].content) if messages else ""
        print(f"Agent: {response}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

    print(f"\nTotal API calls used: {API_CALL_COUNT}")

if __name__ == "__main__":
    main()
