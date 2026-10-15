from enum import StrEnum
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class ENVIRONMENTS(StrEnum):
    """Enumeration of supported deployment environments.

    Members:
        DEVELOPMENT: Local development environment
        PRODUCTION: Live production environment
    """
    DEVELOPMENT = "DEVELOPMENT"
    PRODUCTION = "PRODUCTION"

# Application settings
APP_NAME = os.getenv('APP_NAME', 'Finsight')
ENV = os.getenv('ENV', ENVIRONMENTS.DEVELOPMENT)

# Base directory paths
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')

# Data and storage paths
DATA_URL = os.getenv('DATA_URL', None)
UPLOAD_PATH = os.path.join(BASE_DIR, f"uploads")
DATA_DIR = os.path.join(BASE_DIR, "datasets")
VECTOR_DIR = os.path.join(BASE_DIR, "vectors")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# LLM configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHAT_MODEL = os.getenv('CHAT_MODEL', 'gemini-2.5-flash')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0))

OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
OLLAMA_CHAT_MODEL = os.getenv('OLLAMA_MODEL', 'gemma4:31b-cloud')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')
EMBEDDING_URL = os.getenv('EMBEDDING_MODEL', 'https://qwen-embed.publicaai.com/v1')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')

UPLOAD_LIMIT = int(os.getenv('UPLOAD_LIMIT', 11331600))
UPLOAD_TYPES = {'text/csv', 'application/pdf'}
