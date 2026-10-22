from langchain_ollama import ChatOllama

from src.lib.config import OLLAMA_CHAT_MODEL, TEMPERATURE, OLLAMA_API_KEY, OLLAMA_BASE_URL

def get_ollama_llm(model: str = OLLAMA_CHAT_MODEL, temperature: float = TEMPERATURE) -> ChatOllama:
    """Get LLM instance based on environment configuration.

    Args:
        model: Model name to use
        temperature: Sampling temperature for response generation

    Returns:
        ChatOllama LLM instance configured for environment

    Notes:
        - Uses cloud configuration for PRODUCTION and STAGING
        - Uses local Ollama for DEVELOPMENT and TESTING
    """
    config = {
        'model': model,
        'base_url': OLLAMA_BASE_URL,
        'temperature': temperature,
        'api_key': OLLAMA_API_KEY,
        'cloud': True
    }

    return ChatOllama(**config)
