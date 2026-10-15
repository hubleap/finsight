from langchain_community.tools import StructuredTool
from langgraph.graph import END, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, Optional
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from src.lib.llms import get_llm
from src.lib.knowledge import Knowledge
from src.lib.tool_set import AgentTool
from src.models.analysis import ANALYSIS_DEPTH
from src.models.workflow import BaseWorkflow, WorkflowSettings


ANALYSIS_TYPE = Literal[
    "anomaly_detection",
    "distribution_analysis",
    "trend_analysis",
    "spread_analysis",
    "correlation_analysis"
]


class Insight(BaseModel):
    """Individual analysis insight.

    Attributes:
        classification: Type of analysis performed
        description: Description of the analysis
        evidence: Summary of supporting evidence
        confidence: Confidence score (0-1)
    """
    classification: ANALYSIS_TYPE = Field(..., description="The type of analysis")
    description: str = Field(..., description="The description of the analysis")
    evidence: str = Field(..., description="Summary of relevant retrieved patterns or records")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class Analysis(BaseModel):
    """Complete financial analysis results.

    Attributes:
        insights: List of insights or summary string
        sufficient_data: Whether enough data was available
        recommendation: Actionable recommendation
        confidence: Overall confidence score
    """
    insights: list[Insight] | str = Field(..., description="The analytical insight from the data")
    sufficient_data: bool = Field(..., description="Indicates if enough historical data was available")
    recommendation: str = Field(..., description="Recommendation based on the analysis")
    confidence: float = Field(..., description="The confidence in the analysis", ge=0, le=1)


class AnalysisState(BaseModel):
    """State model for analysis workflow.

    Attributes:
        depth: Analysis depth level
        messages: Conversation history
        analysis: Analysis results
    """
    depth: ANALYSIS_DEPTH = Field("Full", description="The depth of the analysis")
    messages: Annotated[list[AnyMessage], add_messages] = Field([], description="The conversation history")
    analysis: Optional[Analysis | str] = Field(None, description="The analysis in the data")


class AnalysisWorkflow(BaseWorkflow):
    """Workflow for farm operational data analysis using RAG.

    Retrieves farm records from the vector store and produces
    evidence-based insights with recommendations.
    """
    TOOL_MODEL = "tools"

    def __init__(self, user_id: str = None, structured_output: bool = True, settings=WorkflowSettings()):
        """Initialize the analysis workflow.

        Args:
            user_id: User ID for personalized knowledge store
            structured_output: Whether to use structured output
            settings: Workflow configuration settings
        """
        super().__init__(state_schema=AnalysisState, settings=settings)

        self.structured_output = structured_output
        self.set_tools(user_id)
        self._build_graph()

    def set_tools(self, user_id: str = None):
        """Configure tools for the workflow.

        Args:
            user_id: User ID for knowledge manager
        """
        self.tools = []
        if user_id:
            km = Knowledge(user_id)
            self.tools.append(km.retrieve_tool())

        self.tool_model = ToolNode(self.tools)

    def _build_graph(self):
        """Build the workflow graph with analyze, tool, and structure nodes."""
        self.add_node(self.ANALYZE, self.analyze)
        self.add_node(self.TOOL_MODEL, self.tool_model)
        self.add_node(self.STRUCTURE, self.structure)

        self.set_entry_point(self.ANALYZE)
        self.add_conditional_edges(self.ANALYZE, self.tools_condition, [self.TOOL_MODEL, self.STRUCTURE, END])
        self.add_edge(self.TOOL_MODEL, self.ANALYZE)
        self.set_finish_point(self.STRUCTURE)

        self.graph = self.compile(checkpointer=MemorySaver())

    ANALYZE = "analyze"
    def analyze(self, state: AnalysisState) -> dict:
        """Analyze data using LLM with tool access.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with LLM response messages
        """
        prompt = """
            Role
            You are a farm operational data analyst for Rivers & Roots. You evaluate farm behavior by retrieving verified historical records, constructing analytical interpretations, and producing evidence-based insights grounded strictly in observed data.

            Task
            Use only the conversation messages as the input context. No external analysis context is provided. Interpret the user"s request from the messages, retrieve relevant historical farm data using the knowledge_retriever tool, analyze the data to identify meaningful statistical or behavioral patterns, evaluate whether the observed or implied behavior aligns with historical patterns, generate analytical insights supported by retrieved evidence, and produce structured results that match the required schema.

            Rules (Follow strictly)
            - Treat conversation messages as the only source of user-provided information.
            - Always retrieve historical data before making factual or analytical conclusions.
            - Base insights strictly on retrieved evidence.
            - Do NOT infer missing facts, fabricate records, or assume context not present in messages.
            - If retrieved data is insufficient for reliable analysis, indicate this explicitly.
            - An analysis may produce zero, one, or multiple insights.
            - Each insight must include classification, description, evidence, and confidence.
            - Atleast one insight must be generated for each class: anomaly_detection, distribution_analysis, trend_analysis, spread_analysis, correlation_analysis
            - Evidence must reference observable properties of retrieved data (patterns, variation, frequency, magnitude, or relationships).
            - Confidence must reflect strength, relevance, and volume of retrieved data.
            - Keep descriptions objective, analytical, and concise.
            - Output must be valid JSON matching the schema exactly.

            Tools
            You have access to the following tools — use them in order:
            - knowledge_retriever: Retrieves verified farm records from a time-series vector store. Use multiple retrievals if needed to establish patterns, baselines, or trends. Always use this tool before making factual determinations.

            Input
            - {depth}: The depth of the analysis

            Output (MUST be a valid JSON)
            - insights: list (Each classification should be analyzed, min-length=1)
                - classification: anomaly_detection | distribution_analysis | trend_analysis | spread_analysis | correlation_analysis
                - description: str
                - evidence: str
                - confidence: float (0-1)
            - recommendation: str
            - sufficient_data: bool
            - confidence: float (0-1)
        """
        formated_propmt = prompt.format(depth=state.depth)
        system_message = SystemMessage(content=formated_propmt)
        llm = get_llm(temperature=self.settings.temperature, model='gpt-oss:120b-cloud').bind_tools(self.tools)
        response = llm.invoke(
            [system_message] +
            state.messages
        )
        return {'messages': response}

    def tools_condition(self, state: AnalysisState) -> str:
        """Route based on whether tools were called.

        Args:
            state: Current workflow state

        Returns:
            Next node name (TOOL_MODEL, STRUCTURE, or END)
        """
        if not state.messages:
            return END

        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return self.TOOL_MODEL

        return self.STRUCTURE

    STRUCTURE = "structure"
    def structure(self, state: AnalysisState) -> dict:
        """Extract structured analysis from LLM response.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with structured analysis
        """
        prompt = """
            Extract the analysis from the data.

            Input
            {data}

            Output (MUST return a valid JSON)
            - insights: [
                - classification: anomaly_detection | distribution_analysis | trend_analysis | spread_analysis | correlation_analysis"
                - description: str
                - evidence: str
                - confidence: float (0-1)
            ]
            - recommendation: str
            - sufficient_data: bool
            - confidence: float (0-1)
        """
        formated_prompt = prompt.format(data=state.messages[-1].content)
        llm = get_llm()
        structured_llm = llm.with_structured_output(Analysis)
        response = structured_llm.invoke(formated_prompt)
        return {"analysis": response}

    def invoke(self, state: AnalysisState) -> Analysis:
        """Execute the workflow and return analysis.

        Args:
            state: Initial analysis state

        Returns:
            Structured analysis results
        """
        super().invoke(state)
        response = self.graph.invoke(state, config=self.settings.config)
        return response["analysis"]

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
                "Performs extensive farm operational analysis using the user's farm records stored in a vector database."
                "Summarize the output to be user friendly, you can use tables, however, never output a json unless requested by the user."
            ),
            func=lambda query, depth="Summary": self.invoke(
                AnalysisState(messages=[HumanMessage(query)], depth=depth)
            )
        )

        return AgentTool(action="Analyzing", tool=tool)
