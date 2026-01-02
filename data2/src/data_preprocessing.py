"""
Data Preprocessing Module for Online Retail Dataset
Handles loading, cleaning, and transformation of transaction data
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


def load_and_preprocess(file_path: str, sample_size: int = None) -> Tuple[List[List[str]], pd.DataFrame]:
    """
    Load and preprocess the Online Retail dataset.
    
    Args:
        file_path: Path to the Excel file
        sample_size: Optional number of transactions to sample (for testing)
    
    Returns:
        transactions: List of transactions (each transaction is a list of items)
        df_clean: Cleaned dataframe for further analysis
    """
    print(f"Loading dataset from {file_path}...")
    
    # Load the dataset
    df = pd.read_excel(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Data cleaning
    print("\n=== Data Cleaning ===")
    
    # 1. Remove rows with missing values in key columns
    initial_rows = len(df)
    df_clean = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description'])
    print(f"Removed {initial_rows - len(df_clean)} rows with missing InvoiceNo/StockCode/Description")
    
    # 2. Remove cancelled orders (InvoiceNo starting with 'C')
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"Removed cancelled orders, remaining: {len(df_clean)} rows")
    
    # 3. Remove rows with negative or zero quantities
    df_clean = df_clean[df_clean['Quantity'] > 0]
    print(f"Removed negative/zero quantities, remaining: {len(df_clean)} rows")
    
    # 4. Remove non-product items (e.g., postage, discounts)
    # These typically have stock codes starting with letters like POST, D, C, M
    df_clean = df_clean[~df_clean['StockCode'].astype(str).str.match(r'^[A-Z]+$')]
    print(f"Removed non-product items, remaining: {len(df_clean)} rows")
    
    # 5. Clean description text
    df_clean['Description'] = df_clean['Description'].str.strip().str.upper()
    
    # 6. Convert InvoiceDate to datetime
    if 'InvoiceDate' in df_clean.columns:
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    print(f"\nFinal cleaned dataset shape: {df_clean.shape}")
    print(f"Unique invoices: {df_clean['InvoiceNo'].nunique()}")
    print(f"Unique products: {df_clean['Description'].nunique()}")
    
    # Transform to transaction format
    print("\n=== Transforming to Transaction Format ===")
    transactions = df_clean.groupby('InvoiceNo')['Description'].apply(list).values.tolist()
    
    if sample_size and sample_size < len(transactions):
        print(f"Sampling {sample_size} transactions for analysis...")
        np.random.seed(42)
        indices = np.random.choice(len(transactions), sample_size, replace=False)
        transactions = [transactions[i] for i in indices]
    
    print(f"Total transactions: {len(transactions)}")
    print(f"Average items per transaction: {np.mean([len(t) for t in transactions]):.2f}")
    print(f"Max items in a transaction: {max([len(t) for t in transactions])}")
    print(f"Min items in a transaction: {min([len(t) for t in transactions])}")
    
    return transactions, df_clean


def get_item_frequencies(transactions: List[List[str]]) -> Dict[str, int]:
    """Calculate frequency of each item across all transactions."""
    freq = defaultdict(int)
    for transaction in transactions:
        for item in set(transaction):  # Count once per transaction
            freq[item] += 1
    return dict(freq)


def segment_by_country(df: pd.DataFrame, target_country: str = 'United Kingdom') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Segment the dataset by country.
    
    Args:
        df: Cleaned dataframe
        target_country: Country to separate
    
    Returns:
        df_target: DataFrame for target country
        df_others: DataFrame for other countries
    """
    if 'Country' not in df.columns:
        raise ValueError("DataFrame does not have 'Country' column")
    
    df_target = df[df['Country'] == target_country].copy()
    df_others = df[df['Country'] != target_country].copy()
    
    print(f"\n=== Country Segmentation ===")
    print(f"{target_country}: {len(df_target)} rows ({len(df_target)/len(df)*100:.1f}%)")
    print(f"Other countries: {len(df_others)} rows ({len(df_others)/len(df)*100:.1f}%)")
    
    return df_target, df_others


def segment_by_period(df: pd.DataFrame, date_column: str = 'InvoiceDate') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Segment the dataset by time period (first half vs second half).
    
    Args:
        df: Cleaned dataframe
        date_column: Name of the date column
    
    Returns:
        df_period1: DataFrame for first period
        df_period2: DataFrame for second period
    """
    if date_column not in df.columns:
        raise ValueError(f"DataFrame does not have '{date_column}' column")
    
    median_date = df[date_column].median()
    
    df_period1 = df[df[date_column] <= median_date].copy()
    df_period2 = df[df[date_column] > median_date].copy()
    
    print(f"\n=== Time Period Segmentation ===")
    print(f"Period 1 (up to {median_date.date()}): {len(df_period1)} rows")
    print(f"Period 2 (after {median_date.date()}): {len(df_period2)} rows")
    
    return df_period1, df_period2


def print_sample_transactions(transactions: List[List[str]], n: int = 5):
    """Print sample transactions for inspection."""
    print(f"\n=== Sample Transactions (first {n}) ===")
    for i, transaction in enumerate(transactions[:n], 1):
        print(f"Transaction {i}: {len(transaction)} items")
        print(f"  Items: {transaction[:5]}{'...' if len(transaction) > 5 else ''}")


if __name__ == "__main__":
    # Test the preprocessing module
    transactions, df_clean = load_and_preprocess("Online Retail.xlsx", sample_size=5000)
    print_sample_transactions(transactions)
    
    # Get item frequencies
    item_freq = get_item_frequencies(transactions)
    print(f"\nTop 10 most frequent items:")
    for item, freq in sorted(item_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {item}: {freq} transactions")
