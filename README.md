# Building a Full RAG System with Agno: Semantic Chunking, Metadata Filters, and Document Processing

This repo is part of a RAG series of [The Pipe & The Line Substack](https://thepipeandtheline.substack.com/).

This project demonstrates how to build a complete Retrieval-Augmented Generation (RAG) system using [Agno](https://github.com/agno-ai/agno), a powerful Python framework for AI agents and knowledge management. We'll walk through implementing semantic chunking, metadata filtering, and document processing for a social media posts knowledge base.

## 🎯 What We're Building

A RAG system that can:
- Process and chunk social media posts semantically
- Store documents with metadata filters (user_id, tags, reactions, views)
- Query the knowledge base with specific filters
- Provide contextual responses using an AI agent

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   JSON Posts    │───▶│  Document Loader │───▶│  Vector Store   │
│   (knowledge/   │    │  (Semantic       │    │   (PostgreSQL   │
│    files/)      │    │   Chunking)      │    │    + pgvector)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Metadata        │    │  AI Agent       │
                       │  Filters         │    │  (GPT-4 + RAG)  │
                       └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <your-repo>
cd rag-demo
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and database URL

# Or export them if you don't use load_dotenv()

export OPENAI_API_KEY=your_openai_key
export DATABASE_URL=postgresql://user:pass@localhost:5432/rag_demo
```

### 2. Fetch Sample Data

```bash
python knowledge/scripts/save_posts.py
```

This fetches 30 sample social media posts from [DummyJSON](https://dummyjson.com/posts) and saves them as individual JSON files in `knowledge/files/`.

### 3. Load Documents to Knowledge Base

```bash
python knowledge/scripts/load_posts.py
```

This processes all JSON files, creates documents with metadata, and loads them into the vector database.

### 4. Run the RAG Agent

```bash
python agent.py
```

## 🔧 Core Components Deep Dive

### 1. Document Processing with Semantic Chunking

```python
from agno.document.chunking.semantic import SemanticChunking
from agno.embedder.openai import OpenAIEmbedder

chunking_strategy = SemanticChunking(
    embedder=OpenAIEmbedder(id="text-embedding-ada-002"), 
    similarity_threshold=0.3
)
```

**Why Semantic Chunking?**
- Traditional fixed-size chunking can break sentences mid-thought
- Semantic chunking groups related content together based on meaning
- `similarity_threshold=0.3` ensures chunks are semantically coherent

### 2. Metadata and Filtering Strategy

```python
def create_document_metadata(post_data):
    metadata = {
        "tags": post_data.get("tags", []),
        "reactions": post_data.get("reactions", {}),
        "views": post_data.get("views", 0),
    }
    
    filters = {
        "user_id": post_data.get("userId", 0),
    }
    
    return Document(
        content=post_data["body"],
        name=post_data["title"],
        meta_data=metadata,
    ), filters
```

**Key Insights:**
- `metadata`: Stored with the document for context
- `filters`: Used for query-time filtering (e.g., "posts by user_id 97")
- Separate concerns: metadata for context, filters for search

### 3. Vector Database Setup

```python
from agno.vectordb.pgvector import PgVector, SearchType

vector_db = PgVector(
    table_name="posts",
    db_url=DATABASE_URL,
    schema="rag_demo",
    search_type=SearchType.hybrid,  # Combines vector + keyword search
    embedder=OpenAIEmbedder(id="text-embedding-ada-002"),
)
```

**Hybrid Search Benefits:**
- Vector search: Semantic similarity
- Keyword search: Exact matches
- Best of both worlds for comprehensive retrieval

### 4. Knowledge Base Configuration

```python
social_media_knowledge_base = JSONKnowledgeBase(
    path="knowledge/files",
    vector_db=vector_db,
    chunking_strategy=chunking_strategy,
)
```

### 5. Filter Initialization

```python
def initialize_knowledge_filters(knowledge_base):
    if knowledge_base.valid_metadata_filters is None:
        knowledge_base.valid_metadata_filters = set()
    
    knowledge_base.valid_metadata_filters.add("user_id")
    # Add other filter keys as needed
```

**Why This Matters:**
- Agno validates filter keys against known metadata structure
- Prevents invalid filter warnings
- Ensures query performance optimization

## 🔍 Querying with Filters

```python
# Query posts by specific user
agent.print_response(
    "What was her terrible habit?",
    knowledge_filters={"user_id": 97}
)

# This will only search documents where user_id = 97
```

## 🔗 Resources

- [Agno Documentation](https://github.com/agno-ai/agno)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)