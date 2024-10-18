import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FilterSelector, FieldCondition, MatchValue
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain_qdrant import QdrantVectorStore  # Updated import
from tqdm import tqdm
from chunking import process_pdf  # Import the process_pdf function from chunking.py
from qdrant_client import models

# Load ASCII art from file
def load_ascii_art(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load .env file from the correct path
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

def initialize_qdrant():
    print("Freeing my mind.. 🧘")
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('QDRANT_API_KEY')
    qdrant_collection = os.getenv('QDRANT_COLLECTION')

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, prefer_grpc=True, timeout=30)
    
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if qdrant_collection not in collection_names:
            print(f"Collection '{qdrant_collection}' doesn't exist. Creating it now.")
            client.create_collection(
                collection_name=qdrant_collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"Collection '{qdrant_collection}' created.")
        # else:
        #     print(f"Collection '{qdrant_collection}' already exists.")
    except Exception as e:
        print(f"Error while initializing Qdrant: {e}")
        raise

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection,
        embedding=embeddings
    )

    return qdrant, client, qdrant_collection

def prepare_collection(client, collection_name, file_name):
    try:
        delete_vectors_by_source(client, collection_name, file_name)
        # print(f"Collection '{collection_name}' is ready for new vectors.")
        print("I'm ready for new knowledge! 🧠✨")
    except Exception as e:
        print(f"Error preparing collection: {e}")
        raise

def create_collection(client, collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print(f"Created new collection: '{collection_name}'")

def delete_vectors_by_source(client, collection_name, file_name):
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=file_name)
                    )
                ]
            )
        )
    )
    # print(f"Deleted vectors with source: '{file_name}' from collection: '{collection_name}'")

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

    # Get the file name from the file path
    file_name = os.path.basename(file_path)

    # Initialize Qdrant
    qdrant, client, collection_name = initialize_qdrant()

    # First, prepare the collection
    prepare_collection(client, collection_name, file_name)

    # Process PDF using the chunking algorithm from chunking.py
    chapter_chunks = process_pdf(file_path)
    
    # Create documents
    documents = [
        Document(
            page_content=content,
            metadata={
                "source": file_name,
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

    file_name = os.path.basename(file_path)
    print(f"Now I know all {len(documents)} chapters from {file_name} 🙏")    
    print(f"Total processing time: {total_time:.2f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file_path = os.path.join(script_dir, "Business_Central_API_v2.0_Reference_2nd_Edition.pdf")
    process_and_embed_pdf(pdf_file_path)
