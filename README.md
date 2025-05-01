# AutoGen Business Central API Demos

This repository contains a collection of demos showcasing different ways to use AutoGen with the Business Central API.

## Demos Overview

1. **Demo 1 (Simple Agent)**: Uses a simple combination of Assistant Agent and Code Executor Agent (2 Agents)
2. **Demo 2 (Team of Agents)**: Uses a team of specialized agents including Manager, Coder, Executor, and Answer Composer in a SelectorGroupChat
3. **Demo 3 (Agent + Tool)**: Uses a single Assistant Agent with a custom Business Central API Tool
4. **Demo 4 (RAG + Tool)**: Uses a single Assistant Agent with both RAG (Retrieval Augmented Generation) and API Tool capabilities

## Setup

### Prerequisites

- Python 3.9+
- Azure OpenAI resources (API key, endpoint, etc.)
- Business Central instance (for API calls)
- Qdrant vector database (for Demo 4 RAG functionality)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with the following variables:
   ```
   # Azure OpenAI
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_API_BASE=https://your-azure-openai-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2023-07-01-preview
   AZURE_OPENAI_API_MODEL=gpt-4
   AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
   
   # Qdrant (for Demo 4 RAG)
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION=bc_api_docs
   
   # Business Central API
   BC_API_URL=your_bc_api_url
   BC_API_COMPANY_ID=your_company_id
   BC_API_USERNAME=your_username
   BC_API_PASSWORD=your_password
   ```

## Running the Demos

### Using the Streamlit App (Recommended)

The Streamlit app provides a unified interface to run all demos:

```
streamlit run streamlit/app.py
```

### Demo 4: RAG + Tool Features

Demo 4 combines RAG (Retrieval Augmented Generation) with API Tool capabilities:

1. The agent can query a vector database to retrieve relevant Business Central API documentation
2. It uses this knowledge along with the API tool to answer complex questions

#### Embedding Documents for RAG

When using Demo 4 in the Streamlit app:

1. Select "Demo 4: RAG + Tool" from the demo dropdown
2. Open the "Document Embedding (RAG)" expander
3. Either:
   - Upload your own PDF document for indexing, or
   - Use the default Business Central API documentation if available
4. Click "Index Document" to process and embed the document
5. Once indexing is complete, you can ask questions about the document content combined with live API capabilities

## Directory Structure

- `demo1/`: Contains code for Demo 1 (Simple Agent)
- `demo2/`: Contains code for Demo 2 (Team of Agents)
- `demo3/`: Contains code for Demo 3 (Agent + Tool)
- `demo4/`: Contains code for Demo 4 (RAG + Tool)
  - `v02` : Autogen 0.2 version 
  - `v04/`: Autogen 0.4 version and above
  - `embedding/`: Tools for document embedding and RAG functionality
- `streamlit/`: Contains the Streamlit application for running all demos

## Requirements

See `requirements.txt` for a complete list of dependencies. 