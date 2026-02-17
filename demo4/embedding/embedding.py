import os
import time
import dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FilterSelector, FieldCondition, MatchValue
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain_qdrant import QdrantVectorStore  # Updated import
from chunking import process_pdf  # Import the modified process_pdf function
from qdrant_client import models

# Load ASCII art from file
def load_ascii_art(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load .env file from the current directory or parent directories
# Try to load from streamlit directory first, then from root
streamlit_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'streamlit', '.env')
root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), '.env')

if os.path.exists(streamlit_env_path):
    dotenv.load_dotenv(streamlit_env_path)
elif os.path.exists(root_env_path):
    dotenv.load_dotenv(root_env_path)
else:
    dotenv.load_dotenv()  # Fallback to default behavior
script_dir = os.path.dirname(os.path.abspath(__file__)) # Keep for later use

def initialize_qdrant():
    print("Freeing my mind.. 🧘")
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('QDRANT_API_KEY')
    qdrant_collection = os.getenv('QDRANT_COLLECTION')    
    # Use HTTP instead of gRPC to avoid 404 errors with some Qdrant cloud instances
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, prefer_grpc=False, timeout=30, check_compatibility=False)
    
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        collection_exists = qdrant_collection in collection_names
        if not collection_exists:
            print(f"Collection '{qdrant_collection}' doesn't exist. Creating it now.")
            client.create_collection(
                collection_name=qdrant_collection,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE), # Assuming size 1536 and Cosine
            )
            print(f"Collection '{qdrant_collection}' created.")
        # else:
            # print(f"Collection '{qdrant_collection}' already exists.")

        # Ensure the payload index exists for metadata.source
        print(f"Ensuring payload index exists for 'metadata.source' in collection '{qdrant_collection}'...")
        try:
            client.create_payload_index(
                collection_name=qdrant_collection,
                field_name="metadata.source",
                field_schema=models.PayloadSchemaType.KEYWORD # Use PayloadSchemaType for keyword index
            )
            print("Payload index for 'metadata.source' ensured.")
        except Exception as index_error:
            # Index might already exist, which is fine
            if "already exists" in str(index_error).lower() or "duplicate" in str(index_error).lower():
                print("Payload index for 'metadata.source' already exists.")
            else:
                # Re-raise if it's a different error
                raise
            
    except Exception as e:
        print(f"Error while initializing Qdrant or creating index: {e}")
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
                        key="metadata.source",
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

def process_and_embed_pdf(file_path, progress_callback=None):
    """Processes and embeds PDF, passing progress_callback to sub-functions and using it for batch loop."""
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

    # Process PDF using the chunking algorithm from chunking.py, passing callback
    chapter_chunks = process_pdf(file_path, progress_callback=progress_callback)
    
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
    num_batches = len(document_batches)
    
    # Add documents to Qdrant in batches, using callback
    if progress_callback:
        progress_callback(0, num_batches, "Starting document embedding...")
        
    for i, batch in enumerate(document_batches):
        batch_start_time = time.time()
        batch_size = len(batch)
        print(f"Processing batch {i+1}/{num_batches} ({batch_size} documents)...")
        
        try:
            add_documents_to_qdrant(qdrant, batch)
            batch_time = time.time() - batch_start_time
            print(f"Batch {i+1}/{num_batches} completed in {batch_time:.2f} seconds")
        except Exception as e:
            batch_time = time.time() - batch_start_time
            print(f"Error in batch {i+1}/{num_batches} after {batch_time:.2f} seconds: {e}")
            raise
        
        if progress_callback:
            # Pass embedding_batch to signal we're in embedding phase (for image display)
            progress_callback(i + 1, num_batches, f"Embedding batch {i+1}/{num_batches}", embedding_batch=i)
        
        # Add a small delay between batches to avoid rate limiting (except for the last batch)
        if i < num_batches - 1:
            time.sleep(0.5)  # 500ms delay to help avoid rate limits

    print("-" * 50)

    end_time = time.time()  # End timing
    total_time = end_time - start_time  # Calculate total time

    file_name = os.path.basename(file_path)
    print(f"Now I know all {len(documents)} chapters from {file_name} 🙏")    
    print(f"Total processing time: {total_time:.2f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    # Example progress callback for standalone execution
    def print_progress(step, total, description):
        percent = (step / total) * 100 if total > 0 else 0
        print(f"PROGRESS: {description} - {step}/{total} ({percent:.2f}%)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file_path = os.path.join(script_dir, "Business_Central_API_v2.0_Reference_2nd_Edition.pdf")
    process_and_embed_pdf(pdf_file_path, progress_callback=print_progress)
