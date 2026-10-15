from datetime import datetime, timedelta
from typing_extensions import Generic
from pydantic import BaseModel, Field, computed_field
from typing_extensions import Literal, Optional, TypeVar


ANALYSIS_PERIOD = Literal['Hour', 'Week Day', 'Month', 'Month Day', 'Year']
ANALYSIS_DEPTH = Literal['Full', 'Summary', 'Dashboard']


T = TypeVar("T")


class Scanner(BaseModel):
    """Parameters for data analysis scanning.

    Defines the time range, granularity, and scope for financial analysis.

    Attributes:
        start_date: Start date for the analysis period
        stop_date: End date for the analysis period
        period: Time granularity for analysis (Hour, Week Day, Month, etc.)
        depth: Analysis depth (Full, Summary, or Dashboard)
        simulation_id: Optional simulation ID for simulated data analysis
        entity: Entity type to analyze (Sale, Expense, or Business)
    """
    start_date: datetime = Field(datetime.now() - timedelta(days=30), description="The start date for analysis")
    stop_date: datetime = Field(datetime.now(), description="The stop date for analysis")
    period: Optional[ANALYSIS_PERIOD] = Field('Month Day', description="The period to analyze")
    depth: Optional[ANALYSIS_DEPTH] = Field('Dashboard', description="The depth of the analysis")
    simulation_id: Optional[str] = Field(None, description="Id of the simulation the analysis belong to")
    entity: Optional[Literal['Sale', 'Expense', 'Business']] = Field(None, description="The entity to analyze")

    @computed_field
    @property
    def days(self) -> int:
        """Calculate number of days in the analysis period."""
        timeframe = self.stop_date - self.start_date
        return abs(timeframe.days)


class Insight(BaseModel):
    """Container for individual analysis insights.

    Attributes:
        name: Name or title of the insight
        data: Dictionary containing insight data
    """
    name: str = Field(None, description="The name of the insight")
    data: dict = Field(None, description="The data of the insight")


class AnalysisTransaction(BaseModel):
    """Structure for individual transaction data in analysis.

    Attributes:
        item: Item name from the transaction
        unit: Unit of measurement
        quantity: Transaction quantity
        amount: Transaction amount
        category: Transaction category
        timestamp: Transaction timestamp
        vendor: Vendor name
        entity: Transaction entity type
    """
    item: Optional[str] = Field(None, description="The item name of the transaction")
    unit: Optional[str] = Field(None, description="The unit of the transaction")
    quantity: Optional[float] = Field(None, description="The quantity of the transaction")
    amount: Optional[float] = Field(None, description="The amount of the transaction")
    category: Optional[str] = Field(None, description="The category of the transaction")
    timestamp: Optional[datetime] = Field(None, description="The timestamp of the transaction")
    vendor: Optional[str] = Field(None, description="The vendor of the transaction")
    entity: Optional[str] = Field(None, description="The entity of the transaction")


class EntityInsightsAnalysis(BaseModel):
    """Complete analysis results for a specific entity.

    Attributes:
        summary: Summary statistics for the entity
        periods: Periodic analysis breakdown
        items: Item-level analysis
        vendors: Vendor-level analysis
        categories: Category-level analysis
    """
    summary: dict = Field(default_factory=dict, description="The summary analysis of the entity")
    periods: dict = Field(default_factory=dict, description="The periodic analysis of the entity")
    items: dict = Field(default_factory=dict, description="The items analysis of the entity")
    vendors: dict = Field(default_factory=dict, description="The vendors analysis of the entity")
    categories: dict = Field(default_factory=dict, description="The categories analysis of the entity")


class EntityInsightsSummary(BaseModel):
    """Summarized entity insights for dashboard display.

    Attributes:
        summary: Overall summary statistics
        periods: Top performing periods
        items: Top items by performance
        vendors: Top vendors by performance
        categories: Top categories by performance
    """
    summary: dict = Field(default_factory=dict, description="The summary analysis of the business")
    periods: list = Field(default_factory=list, description="The top period of the business")
    items: list = Field(default_factory=list, description="The top items of the business")
    vendors: list = Field(default_factory=list, description="The top vendors of the business")
    categories: list = Field(default_factory=list, description="The top categories analysis of the business")


class BusinessAnalysis(BaseModel):
    """Comprehensive business analysis results.

    Contains detailed analysis of sales and expenses across multiple dimensions.

    Attributes:
        summary: Overall business summary
        top_sales_periods: Best performing sales periods
        top_sales_items: Top selling items
        top_sales_categories: Top selling categories
        top_sales_vendors: Top customers by sales
        top_expenses_periods: Highest expense periods
        top_expenses_items: Most expensive items
        top_expenses_categories: Highest expense categories
        top_expenses_vendors: Top vendors by expense amount
    """
    summary: dict = Field(default_factory=dict, description="The summary analysis of the business")
    top_sales_periods: dict = Field(default_factory=dict, description="The top selling periods of the business")
    top_sales_items: dict = Field(default_factory=dict, description="The top selling items of the business")
    top_sales_categories: dict = Field(default_factory=dict, description="The top selling categories of the business")
    top_sales_vendors: dict = Field(default_factory=dict, description="The top selling customers of the business")
    top_expenses_periods: dict = Field(default_factory=dict, description="The top expensive periods of the business")
    top_expenses_items: dict = Field(default_factory=dict, description="The top expensive items of the business")
    top_expenses_categories: dict = Field(default_factory=dict, description="The top expensive categories of the business")
    top_expenses_vendors: dict = Field(default_factory=dict, description="The top expensive vendors of the business")


class Analysis(BaseModel, Generic[T]):
    """Generic container for analysis results with period comparison.

    Attributes:
        scanner: Analysis parameters and configuration
        current: Analysis results for the current period
        previous: Analysis results for the comparison period
    """
    scanner: Scanner = Field(..., description="The period and insight of the analysis")
    current: T = Field(..., description="The analysis of current window")
    previous: T = Field(..., description="The analysis of previous window")
    