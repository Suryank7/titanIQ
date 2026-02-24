import pandas as pd
from langchain_core.tools import tool

DATA_PATH = "data/titanic.csv"

@tool
def get_dataset_schema() -> str:
    """
    Returns the schema of the Titanic dataset: column names, data types, 
    and non-null counts. Use this to understand the structure of the data before analyzing.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        info = []
        info.append("Titanic Dataset Columns:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notnull().sum()
            total = len(df)
            info.append(f"- {col}: {dtype} (Non-Null Count: {non_null}/{total})")
        
        info.append("\nSample Data (First 3 rows):")
        info.append(df.head(3).to_markdown())
        return "\n".join(info)
    except Exception as e:
        return f"Error reading schema: {str(e)}"
