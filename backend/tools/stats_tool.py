import pandas as pd
from langchain_core.tools import tool
import numpy as np

DATA_PATH = "data/titanic.csv"

@tool
def calculate_statistics(column: str, stat_type: str) -> str:
    """
    Calculates specific statistical summaries for a given numerical or categorical column.
    
    Arguments:
    - column: The name of the column in the Titanic dataset (e.g., 'Age', 'Fare', 'Survived').
    - stat_type: The statistic to compute. Options: 'mean', 'median', 'mode', 'std', 'var', 'min', 'max', 'count', 'missing', 'summary'.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        if column not in df.columns:
            return f"Error: '{column}' is not a valid column."
        
        series = df[column]
        if stat_type == 'mean':
            return f"Mean of {column}: {series.mean()}"
        elif stat_type == 'median':
            return f"Median of {column}: {series.median()}"
        elif stat_type == 'mode':
            return f"Mode of {column}: {series.mode().tolist()}"
        elif stat_type == 'std':
            return f"Standard Deviation of {column}: {series.std()}"
        elif stat_type == 'var':
            return f"Variance of {column}: {series.var()}"
        elif stat_type == 'min':
            return f"Minimum of {column}: {series.min()}"
        elif stat_type == 'max':
            return f"Maximum of {column}: {series.max()}"
        elif stat_type == 'count':
            return f"Total count of non-null values for {column}: {series.count()}"
        elif stat_type == 'missing':
            return f"Missing values in {column}: {series.isnull().sum()}"
        elif stat_type == 'summary':
            return f"Summary of {column}:\n{series.describe().to_string()}"
        else:
            return f"Error: Unexpected stat_type '{stat_type}'."
    except Exception as e:
        return f"Error calculating statistics: {str(e)}"
