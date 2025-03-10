import logging
import uuid
from datetime import datetime

from chromadb.api.models.Collection import Collection
from openai import APIError

from ..services.chroma_service import get_context_with_sources, semantic_search
from ..services.openai_service import (
    contextualize_query,
    generate_response,
)

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation sessions and their history for the RAG system.

    This class handles the creation and management of conversation sessions,
    storing messages, and formatting conversation history for use in prompts.
    Each conversation is stored with a unique session ID and maintains a
    chronological history of messages with their roles (user/assistant).

    Attributes:
        conversations (dict): Dictionary storing conversation histories.
            Key: session_id (str)
            Value: list of message dictionaries containing:
                - role: "user" or "assistant"
                - content: message text
                - timestamp: ISO format timestamp
    """

    def __init__(self):
        # In-memory conversation store
        self.conversations = {}

    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

    def get_conversation_history(
        self, session_id: str, max_messages: int = None
    ) -> list[dict]:
        """Get conversation history for a session"""
        if session_id not in self.conversations:
            return []

        history = self.conversations[session_id]
        if max_messages is not None:
            history = history[-max_messages:]

        return history

    def format_history_for_prompt(self, session_id: str, max_messages: int = 5) -> str:
        """Format conversation history for inclusion in prompts"""
        history = self.get_conversation_history(session_id, max_messages)
        formatted_history = ""

        for msg in history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"

        return formatted_history.strip()


def rag_query(collection: Collection, query: str, n_chunks: int = 2):
    """Perform RAG query: retrieve relevant chunks and generate answer"""
    # Get relevant chunks
    semantic_search_results = semantic_search(
        collection=collection, query=query, n_results=n_chunks
    )
    context, sources = get_context_with_sources(semantic_search_results)

    # Generate response
    response = generate_response(query, context)

    return response, sources, semantic_search_results


def conversational_rag_query(
    conversation_manager: ConversationManager,
    collection: Collection,
    query: str,
    session_id: str,
    n_chunks: int = 2,
):
    """Perform RAG query with conversation history"""
    # Get conversation history
    conversation_history = conversation_manager.format_history_for_prompt(session_id)

    # Handle follow-up questions
    query = contextualize_query(query, conversation_history)
    print("Contextualized Query:", query)

    # Get relevant chunks
    context, sources = get_context_with_sources(
        semantic_search(collection, query, n_chunks)
    )
    print("Context:", context)
    print("Sources:", sources)

    response = generate_response(query, context, conversation_history)

    # Add to conversation history
    conversation_manager.add_message(session_id, "user", query)
    conversation_manager.add_message(session_id, "assistant", response)

    return response, sources


def process_query(collection: Collection, query: str) -> tuple[str, list[str], dict]:
    """Process a query and return response with sources"""
    logger.info(f"Processing query: {query}")
    try:
        response, sources, semantic_search_results = rag_query(collection, query)
        logger.info("Query processed successfully")
        return response, sources, semantic_search_results
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise
