from enum import StrEnum
from typing_extensions import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class ToolSet(StrEnum):
    """Enumeration of available AI agent tools.

    Defines the set of tools that can be used by the AI assistant
    for various tasks like document extraction, analysis, and conversation.
    """

    Enhancer = 'Enhancer'
    Searcher = 'Searcher'
    FinancialAnalyst = 'FinancialAnalyst'


class AgentTool(BaseModel):
    """Represents a tool available to an AI agent.

    Attributes:
        tool: The LangChain StructuredTool instance
        action: Optional action identifier for the tool
    """

    tool: StructuredTool
    action: Optional[str]
