"""Data loading utilities for the AllergyP package."""

import pandas as pd
from pathlib import Path
from typing import Union, Optional


def load_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments to pass to pandas.read_csv
        
    Returns:
        pandas.DataFrame: The loaded data
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif filepath.suffix.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def preprocess_data(data: pd.DataFrame, drop_na: bool = False, 
                    columns_to_keep: Optional[list] = None) -> pd.DataFrame:
    """
    Preprocess a dataframe for analysis.
    
    Args:
        data: Input dataframe
        drop_na: Whether to drop rows with NA values
        columns_to_keep: List of columns to keep (None means keep all)
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Keep only specified columns if provided
    if columns_to_keep is not None:
        df = df[columns_to_keep]
    
    # Handle missing values
    if drop_na:
        df = df.dropna()
    
    return df 