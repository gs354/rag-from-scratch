import logging

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from ..config.config import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def get_prompt(context: str, query: str):
    """Generate a prompt combining context and query"""
    prompt = f"""Based on the following context, please provide a relevant 
    and contextual response. If the answer cannot be derived from the context, 
    only say "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Human: {query}

    Assistant:"""

    return prompt


def get_prompt_with_history(context: str, conversation_history: str, query: str):
    """Generate a prompt combining context, history, and query"""
    prompt = f"""Based on the following context and conversation history, 
    please provide a relevant and contextual response. If the answer cannot 
    be derived from the context, only use the conversation history or say 
    "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""

    return prompt


def generate_response(
    query: str, context: str, conversation_history: str | None = None
) -> str:
    """Generate a response using OpenAI"""
    if conversation_history is None:
        prompt = get_prompt(context, query)
    else:
        prompt = get_prompt_with_history(context, conversation_history, query)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        return response.choices[0].message.content
    except RateLimitError as e:
        error_msg = "Rate limit exceeded. Please try again later."
        logger.error(f"{error_msg}: {str(e)}")
        return error_msg
    except APITimeoutError as e:
        error_msg = "Request timed out. Please try again."
        logger.error(f"{error_msg}: {str(e)}")
        return error_msg
    except APIError as e:
        error_msg = "API error occurred. Please try again later."
        logger.error(f"{error_msg}: {str(e)}")
        return error_msg


def contextualize_query(query: str, conversation_history: str):
    """Convert follow-up questions into standalone queries"""
    contextualize_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone 
    question which can be understood without the chat history. Do NOT answer 
    the question, just reformulate it if needed and otherwise return it as is."""

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": contextualize_prompt},
                {
                    "role": "user",
                    "content": f"Chat history:\n{conversation_history}\n\nQuestion:\n{query}",
                },
            ],
        )
        return completion.choices[0].message.content
    except RateLimitError as e:
        error_msg = "Rate limit exceeded. Please try again later."
        logger.error(f"{error_msg}: {str(e)}")
        return error_msg
    except APITimeoutError as e:
        error_msg = "Request timed out. Please try again."
        logger.error(f"{error_msg}: {str(e)}")
        return error_msg
    except APIError as e:
        error_msg = "API error occurred. Please try again later."
        logger.error(f"{error_msg}: {str(e)}")
        return error_msg
