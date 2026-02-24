import pandas as pd
from langchain_core.tools import tool

DATA_PATH = "data/titanic.csv"

@tool
def execute_pandas_query(query: str, explanation: str) -> str:
    """
    Executes a pandas query on the Titanic dataset (loaded as `df`).
    The query must be a valid Python expression returning a string, number, or small dataframe slice.
    For example: "df['Age'].mean()", "len(df[df['Survived']==1])", or "df.groupby('Sex')['Survived'].mean().to_string()".
    Provide an explanation of what the query is intended to do.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        # Sandbox for eval
        local_vars = {"df": df, "pd": pd}
        result = eval(query, {}, local_vars)
        return f"Result for '{explanation}':\n{result}"
    except Exception as e:
        return f"Error executing query: {str(e)}"
