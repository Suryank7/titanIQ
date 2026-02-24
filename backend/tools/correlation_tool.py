import pandas as pd
from langchain_core.tools import tool

DATA_PATH = "data/titanic.csv"

@tool
def analyze_correlation(column1: str, column2: str) -> str:
    """
    Computes the Pearson correlation coefficient between two continuous numerical columns in the Titanic dataset.
    This reveals if two columns move together.
    
    Arguments:
    - column1: A numerical column name (e.g., 'Age', 'Fare', 'Survived', 'Pclass').
    - column2: Another numerical column name.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        if column1 not in df.columns or column2 not in df.columns:
            return f"Error: Ensure both '{column1}' and '{column2}' exist in the dataset."
        
        # Select only valid numbers
        s1 = pd.to_numeric(df[column1], errors='coerce')
        s2 = pd.to_numeric(df[column2], errors='coerce')
        
        corr = s1.corr(s2)
        return f"The Pearson correlation between {column1} and {column2} is {corr:.4f}."
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"
