from langchain_tavily import TavilySearch
from langgraph.graph import END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool
from typing_extensions import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.lib.knowledge import Knowledge
from src.lib.tool_set import AgentTool
from src.models.workflow import BaseWorkflow, WorkflowSettings
from src.lib.llms import get_ollama_llm
from src.lib.config import APP_NAME


class DocumentRank(BaseModel):
    """Ranking for a single document.

    Attributes:
        index: Position of the document in the results list
        relevance: Relevance score (0-1) to the query
    """
    index: int = Field(description='The index of the document in the list of documents')
    relevance: float = Field(description='The relevance of the document with respect to the user query')


class Ranking(BaseModel):
    """Complete ranking results for retrieved documents.

    Attributes:
        ranks: List of document rankings
        confidence: Overall confidence in the ranking
    """
    ranks: List[DocumentRank] = Field(description='List of ranks for each document with respect to the user query')
    confidence: float = Field(description='The confidence in the ranking of the documents', ge=0, le=1)


class SearchState(BaseModel):
    """State model for search workflow.

    Attributes:
        query: User's search query
        documents: Retrieved documents
        search_count: Number of documents to retrieve
    """
    query: str = Field(..., description='The query of the user')
    search_count: int = Field(default=3, description='The number of documents to retreive')
    documents: Optional[list[Document]] = Field(default_factory=list, description='The retreived documents')


class SearchWorkflow(BaseWorkflow):
    """Workflow for searching documents from vector store or web.

    Supports both local knowledge base search and web search via Tavily.
    Includes document ranking based on relevance.
    """

    def __init__(self, collection_names: list[str] = [], settings=WorkflowSettings(), benchmark: float = 0):
        """Initialize the search workflow.

        Args:
            drive: The drive containing the vector store
            collection_names: List of vector store collection names
            settings: Workflow configuration settings
            benchmark: Minimum relevance threshold for documents
        """
        super().__init__(SearchState, settings=settings)
        self._build_graph()

        self.knowledges = [Knowledge(collection_name=name) for name in collection_names]
        self.settings.online = settings.online or bool(not len(collection_names))
        self.benchmark = benchmark

    def _build_graph(self):
        """Build the workflow graph with retrieve, rank, and web search nodes."""
        self.add_node(self.RETRIEVE, self.retreive)
        self.add_node(self.RANK, self.rank)
        self.add_node(self.WEB_SEARCH, self.web_search)

        self.set_conditional_entry_point(self.should_search_web, [self.RETRIEVE, self.WEB_SEARCH])
        self.add_conditional_edges(self.RETRIEVE, self.should_rank, [self.RANK, END])
        self.add_conditional_edges(self.WEB_SEARCH, self.should_rank, [self.RANK, END])
        self.add_edge(self.RANK, END)

        self.graph = self.compile()

    RETRIEVE = 'retreive'
    def retreive(self, state: SearchState) -> dict:
        """Retrieve documents from knowledge base.

        Args:
            state: Current workflow state with query

        Returns:
            Dictionary with retrieved documents
        """
        retrieved_docs = []

        for knowledge in self.knowledges:
            docs = knowledge.retrieve(state.query, state.search_count)
            retrieved_docs.extend(docs)

        self.logger.info('Retrieved Documents')
        return {'documents': retrieved_docs}

    RANK = 'rank'
    def rank(self, state: SearchState) -> dict:
        """Rank retrieved documents by relevance.

        Args:
            state: Current workflow state with documents

        Returns:
            Dictionary with ranked documents
        """
        prompt = """
            You are the Knowledge Ranking Agent for {app_name}, an intelligent system that retrieves and evaluates financial documents to determine which are most likely to answer a user's query.

            Task
            Your job is to analyze each document's relevance and assign a probability score representing the likelihood that it can accurately and completely answer the given query.


            Rules
            - Assess and Analyze each document individually.
            - Consider content relevance, factual alignment, completeness, and specificity to the query.
            - Output a probability score between 0 and 1, where:
                1.0 → The document almost certainly answers the query.
                0.5 → The document might partially answer or provide related context.
                0.0 → The document is unrelated or unhelpful.
            - Do not include commentary or reasoning outside the JSON response.
            - You must rank each document and make sure the returned list matches the list of documents

            Input
            - {query}: The query of the user
            - {documents}: The retrieved documents

            Output
            - ranks (The ranks of all the retrieved documents):
                - index: "int" (The index of the document in the list)
                - relevance: "float" (a float between 0.0 and 1.0, indicating confidence in the document's ability to answer the query.)
            - confidence (The confidence in the ranking)
        """

        formatted_prompt = prompt.format(query=state.query, documents=state.documents)
        system_message = SystemMessage(content=formatted_prompt)

        llm = get_ollama_llm()
        response = llm.with_structured_output(Ranking).invoke([system_message]+[HumanMessage(content='Rank the documents')])

        ranks = {r.index: r.relevance for r in response.ranks}
        documents = [
            Document(page_content=doc.page_content, metadata={**doc.metadata, 'rank': ranks.get(i, 0)})
            for i, doc in enumerate(state.documents) if ranks.get(i, 0) >= self.benchmark
        ]

        self.logger.info(f'Ranked Documents with confidence: {response.confidence}')
        return {'documents': documents}

    WEB_SEARCH = 'web_search'
    def web_search(self, state: SearchState) -> dict:
        """Search the web for relevant documents.

        Args:
            state: Current workflow state with query

        Returns:
            Dictionary with web search results as documents
        """
        prompt = """
            You are a search optimization assistant.
            Your task is to take a user's question and rewrite it into the most effective, concise web search_query.

            Guidelines:
            1. Keep the core intent of the query intact.
            2. Remove unnecessary filler words like "can you tell me" or "how do you think".
            3. Use keywords and phrases that would retrieve the most relevant search results.
            4. If the question is vague, infer the most likely specific search intent.
            5. Ensure the query is clear, precise, and ready for a search engine.

            Example:
            Query: "What are the best feeding practices for catfish in earthen ponds?"
            Search_Query: "catfish feeding practices earthen ponds FCR"

            Now convert this query into a web search_query:
            Query: {query}
            
            Output: <search_query>
        """
        formatted_prompt = prompt.format(query=state.query)

        llm = get_ollama_llm()
        response = llm.invoke([SystemMessage(content=formatted_prompt)] + [HumanMessage(content=f"Turn the question into a search query")])

        tavily_search = TavilySearch(max_results=state.search_count, search_depth='advanced')
        search_result = tavily_search.invoke(response.content)
        
        documents = [
            Document(page_content=result['content'], metadata={'url': result['url'], 'title': result['title']})
            for result in search_result['results']
        ]
        self.logger.info('Searched Web')
        return {'documents': documents}

    def should_rank(self, state: SearchState) -> str:
        """Determine if ranking should be performed.

        Args:
            state: Current workflow state

        Returns:
            Next node name (RANK or END)
        """
        return self.RANK if self.benchmark else END

    def should_search_web(self, state: SearchState) -> str:
        """Determine if web search should be performed.

        Args:
            state: Current workflow state

        Returns:
            Next node name (WEB_SEARCH or RETRIEVE)
        """
        return self.WEB_SEARCH if self.settings.online else self.RETRIEVE

    def invoke(self, state: SearchState) -> list[Document]:
        """Execute the workflow and return documents.

        Args:
            state: Initial search state

        Returns:
            List of retrieved and ranked documents
        """
        super().invoke(state)
        response = self.graph.invoke(state)
        return response['documents']

    def as_tool(self, name: str, description: str) -> AgentTool:
        """Convert workflow into a LangChain tool.

        Args:
            name: Tool name

        Returns:
            AgentTool wrapping the workflow
        """
        tool = StructuredTool.from_function(
            name=name,
            description=description,
            func=lambda query, search_count=3: self.invoke(
                SearchState(query=query, search_count=search_count)
            )
        )

        return AgentTool(action='Searching', tool=tool)
