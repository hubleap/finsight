from typing import Optional

import pandas as pd
from datetime import datetime

def get_transaction_in_timeframe(df: pd.DataFrame, start: Optional[datetime] = None, stop: Optional[datetime] = None):
    df = df.copy()
    if start is not None:
        df =  df[df['transaction_date'] >= start]

    if stop is not None:
        df =  df[df['transaction_date'] <= stop]
    return df

def get_total_income(df: pd.DataFrame, start: Optional[datetime] = None, stop: Optional[datetime] = None):
    """
        Get the total income based on timeframe
        df: The dataframe
        start: The start date
        stop: The stop date
    """
    df = get_transaction_in_timeframe(df, start, stop)
    df = df[df['transaction_type'] == "Income"]
    return df["amount_ngn"].sum()

def get_total_expense(df: pd.DataFrame, start: Optional[datetime] = None, stop: Optional[datetime] = None):
    """
        Get the total income based on timeframe
        df: The dataframe
        start: The start date
        stop: The stop date
    """

    df = get_transaction_in_timeframe(df, start, stop)
    df = df[df['transaction_type'] == "Expense"]
    return df["amount_ngn"].sum()

def get_expense_categories(df: pd.DataFrame, start: Optional[datetime] = None, stop: Optional[datetime] = None):
    """
        Get the total income based on timeframe
        df: The dataframe
        start: The start date
        stop: The stop date
    """
    df = get_transaction_in_timeframe(df, start, stop)
    df = df[df['transaction_type'] == "Expense"]
    return df["category"].unique()

def get_net_cashflow(df: pd.DataFrame, start: Optional[datetime] = None, stop: Optional[datetime] = None):
    """
        Get the total income based on timeframe
        df: The dataframe
        start: The start date
        stop: The stop date
    """
    total_income = get_total_income(df, start, stop)
    total_expense = get_total_expense(df, start, stop)
    return total_income - total_expense

def get_top_income(df: pd.DataFrame, column: str, start: Optional[datetime] = None, stop: Optional[datetime] = None, n=5):
    """
        Get the total income based on timeframe
        df: The dataframe
        start: The start date
        stop: The stop date
    """
    df = get_transaction_in_timeframe(df, start, stop)
    df = df[df['transaction_type'] == "Income"]
    columns = df.groupby(column)['amount_ngn'].sum()
    return columns.head(n)

def get_golden_dataset(df: pd.DataFrame, start: Optional[datetime] = None, stop: Optional[datetime] = None):
    total_income = get_total_income(df, start=start, stop=stop)
    total_expense = get_total_expense(df, start=start, stop=stop)
    expense_categories = get_expense_categories(df, start=start, stop=stop)
    net_cashflow = get_net_cashflow(df, start=start, stop=stop)
    top_income_categories = get_top_income(df, 'category', start=start, stop=stop)

    # How much did the business make
    dataset = [
        {
            'question': "How mush did the business make?",
            'answer': total_income
        },
        {
            'question': "How mush did the business spend?",
            'answer': total_expense
        },
        {
            'question': "What did the business spend money on?",
            'answer': expense_categories
        },
        {
            'question': "What is the net cashflow",
            'answer': net_cashflow
        },
        {
            'question': "Which category generates the most revenue",
            'answer': top_income_categories
        },
    ]

    return pd.DataFrame(dataset)