from dotenv import load_dotenv
import os
from textwrap import dedent
from knowledge.scripts.valid_filters import initialize_knowledge_filters

from agno.agent import Agent
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.json import JSONKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.semantic import SemanticChunking


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

instructions = dedent(
    """
    You are a helpful assistant that can answer questions about the social media posts.
    You can also use the knowledge base to answer questions.
"""
)

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

agent = Agent(
    name="Social Media Agent",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    knowledge=social_media_knowledge_base,
    instructions=instructions,
    search_knowledge=True,
    markdown=True,
)

initialize_knowledge_filters(social_media_knowledge_base)

agent.print_response(
    "What was her terrible habit?",
    knowledge_filters={"user_id": 97},  # knowledge_filters={"user_id": 19},
)
