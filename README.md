# Business Central API Agent Demos

This repository contains a collection of demos showcasing different ways to use AI agent frameworks (Autogen and Microsoft Agent Framework) with the Business Central API.

## Demos Overview

Each demo is available in **two frameworks**:
- **Autogen v0.4**: The established agent framework (autogen_agentchat package)
- **Agent Framework v1.0**: Microsoft's new Agent Framework (Python SDK, beta)

### Demo 1: Simple Agent (Code + Executor)
- **Autogen**: Uses AssistantAgent + CodeExecutorAgent (2 agents)
- **Agent Framework**: Uses single agent with custom execute_python_code tool
- **Use Case**: Generate Python code to call BC API, execute it locally, and return results

### Demo 2: Team of Agents
- **Autogen**: Uses SelectorGroupChat with Coder, Executor, and Answer Composer (3 agents)
- **Agent Framework**: Uses GroupChatBuilder with Coder and Answer Composer + Orchestrator (2 agents + orchestrator)
- **Use Case**: Multi-agent collaboration for complex BC API workflows

### Demo 3: Agent + Tool
- **Autogen**: Uses AssistantAgent with custom FunctionTool (1 agent)
- **Agent Framework**: Uses agent with @tool decorator (1 agent)
- **Use Case**: Direct API calls with structured tool parameters

### Demo 4: RAG + Tool
- **Autogen**: Uses AssistantAgent with RAG and API tool (1 agent)
- **Agent Framework**: Uses agent with RAG (@tool) and API tool (1 agent)
- **Use Case**: Retrieval-Augmented Generation from BC documentation + live API calls

## Setup

### Prerequisites

- Python 3.9+
- Azure OpenAI resources (API key, endpoint, deployment names)
- Business Central instance with OAuth 2.0 app registration
- Qdrant vector database (for Demo 4 RAG functionality)

### Frameworks Used

- **Autogen v0.4**: `autogen_agentchat` and `autogen-ext[openai]`
- **Agent Framework v1.0**: `agent-framework` (pre-release, beta)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the `streamlit/` directory with the following variables:
   ```env
   # Azure OpenAI
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2023-07-01-preview
   AZURE_OPENAI_API_MODEL=gpt-4o
   AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
   
   # Qdrant (for Demo 4 RAG)
   QDRANT_URL=https://your-qdrant-cluster.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION=bc_api_docs
   
   # Business Central API (OAuth 2.0 Client Credentials)
   BC_CLIENT_ID=your_azure_ad_client_id
   BC_CLIENT_SECRET=your_azure_ad_client_secret
   BC_TENANT_ID=your_azure_ad_tenant_id
   BC_COMPANY_ID=your_bc_company_guid
   BC_ENVIRONMENT=sandbox
   ```
   
   **Note**: Both Autogen and Agent Framework read from the same `.env` file. The Agent Framework uses these variables with explicit parameter passing.

## Running the Demos

### Using the Streamlit App (Recommended)

The Streamlit app provides a unified interface to run all demos with both frameworks:

```bash
python3 -m venv .venv                 # Create virtual environment
source .venv/bin/activate             # Activate (use .venv\Scripts\activate on Windows)
pip install -r requirements.txt       # Install dependencies
streamlit run streamlit/app.py        # Run the app
```

In the Streamlit UI:
1. Use the **Framework** radio button to switch between Autogen v0.4 and Agent Framework v1.0
2. Select a **Demo** from the dropdown
3. Enter your prompt and interact with the agent(s)

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

```
demo1/
  autogen/
    v0.2/          # Autogen 0.2 implementation
    v0.4/          # Autogen 0.4 implementation
  agent-framework/
    v1.0/          # Agent Framework implementation
demo2/
  autogen/
    v0.2/
    v0.4/
  agent-framework/
    v1.0/
demo3/
  autogen/
    v0.2/
    v0.4/
  agent-framework/
    v1.0/
demo4/
  autogen/
    v0.2/
    v0.4/
  agent-framework/
    v1.0/
  embedding/       # Shared RAG tools (used by all frameworks)
streamlit/         # Unified UI for all demos and frameworks
```

## Requirements

See `requirements.txt` for a complete list of dependencies. 