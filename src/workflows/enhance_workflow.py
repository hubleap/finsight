from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import Optional
from langchain_core.tools import StructuredTool

from src.lib.tool_set import AgentTool
from src.models.workflow import BaseWorkflow, WorkflowSettings
from src.lib.llms import get_llm


class EnhanceState(BaseModel):
    """State model for query enhancement workflow.

    Attributes:
        query: Original user query
        context: Conversation history for context
        guide: Optional enhancement guide
        enhanced: Resulting enhanced query
    """
    query: str = Field(..., description="The user query")
    context: list = Field(..., description="The conversation history")
    guide: Optional[str] = Field(None, description="A guide to make enhancement better")
    enhanced: Optional[str] = Field(None, description="The enhanced query")


class EnhanceWorkflow(BaseWorkflow):
    """Workflow for enhancing and clarifying user queries.

    Rewrites queries to be clearer and more specific using
    conversation context.
    """

    def __init__(self, settings=WorkflowSettings()):
        """Initialize the enhance workflow.

        Args:
            settings: Workflow configuration settings
        """
        super().__init__(EnhanceState, settings=settings)
        self._build_graph()

    def _build_graph(self):
        """Build the workflow graph with enhancement node."""
        self.add_node(self.ENHANCE, self.enhance)
        self.set_entry_point(self.ENHANCE)
        self.set_finish_point(self.ENHANCE)
        self.graph = self.compile()

    ENHANCE = "enhance"
    def enhance(self, state: EnhanceState) -> dict:
        """Enhance a query using context and optional guide.

        Args:
            state: Current workflow state with query and context

        Returns:
            Dictionary with enhanced query
        """
        prompt = """
            Role
            You are the Query Enhancer.

            Objective
            Rewrite the user’s query to be clearer, more specific, and fully self-contained, without changing its original meaning.

            Guidelines
            - Use the context to improve the query
            - Optional guide may be provided.
            - Use them only if they improve accuracy or clarity.
            - Keep the language simple, direct, and easy to understand.
            - Do not add new intent, assumptions, or extra information.

            Input
            - User query - {query}
            - Conversation history - {context}
            - A guide to make enhancement better {guide}

            Output
            - the improved version of the original query
        """

        formatted_prompt = prompt.format(context=state.context, guide=state.guide, query=state.query)
        print(formatted_prompt)
        system_message = SystemMessage(content=formatted_prompt)

        llm = get_llm()
        response = llm.invoke([system_message]+[HumanMessage(content="Enhance the query")])
        return {"enhanced": response.content}

    def invoke(self, state: EnhanceState) -> str:
        """Execute the workflow and return enhanced query.

        Args:
            state: Initial enhance state

        Returns:
            Enhanced query string
        """
        super().invoke(state)
        response = self.graph.invoke(state)
        return response["enhanced"]

    def as_tool(self, name: str) -> AgentTool:
        """Convert workflow into a LangChain tool.

        Args:
            name: Tool name

        Returns:
            AgentTool wrapping the workflow
        """
        tool = StructuredTool.from_function(
            name=name,
            description=(
                "Rewrite a user query to be clearer, more specific, and self-contained without changing its original intent."
            ),
            func=lambda query, context, guide=None: self.invoke(
                EnhanceState(query=query, context=context, guide=guide)
            )
        )

        return AgentTool(action="Enhancing", tool=tool)