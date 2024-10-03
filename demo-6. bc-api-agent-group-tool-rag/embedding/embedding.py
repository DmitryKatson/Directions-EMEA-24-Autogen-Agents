import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain_qdrant import QdrantVectorStore  # Updated import
from tqdm import tqdm
from chunking import process_pdf  # Import the process_pdf function from chunking.py

# Load ASCII art from file
def load_ascii_art(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load .env file from the correct path
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

def initialize_qdrant():
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('QDRANT_API_KEY')
    qdrant_collection = os.getenv('QDRANT_COLLECTION')

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, prefer_grpc=True, timeout=30)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    qdrant = QdrantVectorStore(  # Updated to QdrantVectorStore
        client=client,
        collection_name=qdrant_collection,
        embedding=embeddings
    )

    return qdrant, client, qdrant_collection

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def add_documents_to_qdrant(qdrant, documents):
    qdrant.add_documents(documents=documents)

def process_and_embed_pdf(file_path):
    start_time = time.time()  # Start timing

    # Load and display ASCII art
    ascii_art_path = os.path.join(os.path.dirname(__file__), 'ascii-art.txt')
    ascii_art = load_ascii_art(ascii_art_path)
    print(ascii_art)
    print("API Guru Agent is starting to read the document...")
    print("Beep boop! 🤖📚")
    print("-" * 50)

    # Initialize Qdrant
    qdrant, client, collection_name = initialize_qdrant()

    # First, delete the collection if it exists
    try:
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )        
    except Exception as e:
        print(f"Error: {e}")

    # Process PDF using the chunking algorithm from chunking.py
    chapter_chunks = process_pdf(file_path)
    
    # Create documents
    documents = [
        Document(
            page_content=content,
            metadata={
                "source": file_path,
                "chapter": title
            }
        ) for title, content in chapter_chunks.items()
    ]
    
    # Split documents into batches of 10
    batch_size = 10
    document_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    
    # Add documents to Qdrant in batches
    for i, batch in enumerate(tqdm(document_batches, desc=f"Embedding docs: {len(documents)}", unit="batch")):
        add_documents_to_qdrant(qdrant, batch)

    print("-" * 50)

    end_time = time.time()  # End timing
    total_time = end_time - start_time  # Calculate total time

    print(f"Finished embedding {len(documents)} chapters from {file_path}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file_path = os.path.join(script_dir, "Business_Central_API_v2.0_Reference_2nd_Edition.pdf")
    process_and_embed_pdf(pdf_file_path)
