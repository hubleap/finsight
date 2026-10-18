from typing_extensions import Literal, Optional
from uuid import uuid4
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field, computed_field

from src.lib.logger import get_logger
from src.models.document import UploadDocument


class CreateChat(BaseModel):
    """Model for creating a new chat message.

    Attributes:
        content: User's query or message content
        session_id: Unique identifier for the chat session
        role: Message role (user or assistant)
        document: Optional uploaded document attached to the message
        reasoning: Reasoning level for the response (0-1)
    """
    content: str = Field(..., description="User query")
    session_id: str = Field(str(uuid4()), description='The session id')
    role: Literal['user', 'assistant'] = Field('user')
    document: Optional[UploadDocument] = Field(None, description='The base64 uploaded document')
    reasoning: Optional[float] = Field(0, description='The reasoning level required')


class Conversation(BaseModel):
    """Model for storing conversation history.

    Attributes:
        query: User's question or message
        reply: Assistant's response
    """
    query: str = Field(..., description="User query")
    reply: str = Field(..., description="Bot reply")


class Chat(BaseModel):
    """Model for individual chat messages stored in database.

    Attributes:
        content: Message content
        session_id: Session identifier
        role: Message role (user or assistant)
    """
    content: str = Field(...)
    session_id: str = Field(...)
    role: Literal['user', 'assistant']


class WorkflowSettings(BaseModel):
    """Settings for AI workflow execution.
    Attributes:
        session_id: Unique identifier for the workflow session
        temperature: LLM temperature setting for response generation
    """
    session_id: str = Field(default=uuid4().hex, description='The session id of the workflow')
    temperature: float = Field(default=0, description='The temperature of the workflow')
    online: bool = Field(default=True, description='Whether the workflow is online or offline')

    @computed_field
    @property
    def config(self) -> dict:
        """Generate LangGraph configuration dictionary with thread ID.

        Returns:
            Configuration dict for LangGraph thread management
        """
        return {'configurable': {'thread_id': self.session_id}}


class BaseWorkflow(StateGraph):
    """Base class for AI workflows using LangGraph.

    Provides common workflow functionality for AI agent implementations.
    """

    def __init__(self, state_schema: type, settings = WorkflowSettings()) -> None:
        """Initialize the workflow with state schema and settings.

        Args:
            state_schema: Pydantic model defining workflow state
            settings: Workflow configuration settings
        """
        super().__init__(state_schema)
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)

    def invoke(self, state):
        """Invoke the workflow with given state.

        Args:
            state: Current workflow state
        """
        self.logger.info('Invoked')


class BaseAgent(BaseWorkflow):
    """Base class for AI agents.

    Extends BaseWorkflow with streaming capabilities for responses.
    """

    def __init__(self, state_schema: type, settings = WorkflowSettings()) -> None:
        """Initialize the agent with state schema and settings.

        Args:
            state_schema: Pydantic model defining agent state
            settings: Agent configuration settings
        """
        super().__init__(state_schema)
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)

    def stream(self, state, db, cache):
        """Stream responses from the agent.

        Args:
            state: Current agent state
            db: Database connection
            cache: Cache connection
        """
        ...