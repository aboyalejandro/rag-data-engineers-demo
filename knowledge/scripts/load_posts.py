import os
import json

from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List, Tuple

from agno.document import Document
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.json import JSONKnowledgeBase
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.semantic import SemanticChunking


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

social_media_knowledge_base = JSONKnowledgeBase(
    path="knowledge/files",
    vector_db=PgVector(
        table_name="posts",
        db_url=DATABASE_URL,
        schema="rag_demo",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(
            id="text-embedding-ada-002",
        ),
    ),
    chunking_strategy=SemanticChunking(
        embedder=OpenAIEmbedder(id="text-embedding-ada-002"), similarity_threshold=0.3
    ),
)


def create_document_metadata(
    post_data: Dict[str, Any],
) -> Tuple[Document, Dict[str, Any]]:
    """Create a knowledge document from post data."""

    metadata = {
        "tags": post_data.get("tags", []),
        "reactions": post_data.get("reactions", {}),
        "views": post_data.get("views", 0),
    }

    filters = {
        "user_id": post_data.get("userId", 0),
    }

    return (
        Document(
            content=post_data["body"],
            name=post_data["title"],
            meta_data=metadata,
        ),
        filters,
    )


def generate_documents() -> List[Tuple[Document, Dict[str, Any]]]:
    """Generate documents from all JSON posts in knowledge/files directory."""
    files_dir = Path("knowledge/files")
    documents = []

    # Get all JSON files
    json_files = list(files_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files to process")

    for file_path in json_files:
        try:
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                post_data = json.load(f)

            # Create document
            document, filters = create_document_metadata(post_data)
            documents.append((document, filters))

            print(
                f"✓ Generated document: {file_path.name} - {post_data.get('title', 'No title')}"
            )

        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {str(e)}")

    print(f"Generated {len(documents)} documents")
    return documents


def load_posts():
    """Load all generated documents to the knowledge base."""
    doc_filter_pairs = generate_documents()

    if doc_filter_pairs:
        try:
            for doc, filters in doc_filter_pairs:
                social_media_knowledge_base.load_documents(
                    documents=[doc],
                    filters=filters,
                )
            print(
                f"✓ Successfully loaded {len(doc_filter_pairs)} documents to knowledge base"
            )
        except Exception as e:
            print(f"✗ Error loading documents to knowledge base: {str(e)}")
    else:
        print("No documents to load")


if __name__ == "__main__":
    load_posts()
