import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
from typing import Optional
from langchain_core.tools import tool

DATA_PATH = "data/titanic.csv"
CHART_DIR = "frontend/charts"

if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

@tool
def generate_visualization(chart_type: str, x_column: str, y_column: Optional[str] = None, hue_column: Optional[str] = None, title: Optional[str] = None) -> str:
    """
    Generates a chart and saves it as an image. Use this tool when the user asks to "plot", "graph", "visualize" or "show me a chart".
    Supported chart_type: 'histogram', 'bar', 'pie', 'box', 'violin', 'scatterplot', 'heatmap'.
    Arguments:
    - chart_type: Type of chart.
    - x_column: The primary column for the X-axis (required).
    - y_column: The secondary column for the Y-axis (optional, required for scatterplot/bar/box/violin).
    - hue_column: A grouping column for color coding (optional).
    - title: Title of the chart.
    Returns the file path of the saved chart to be displayed in the UI.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        if chart_type == 'histogram':
            sns.histplot(data=df, x=x_column, hue=hue_column, kde=True, bins=30)
        elif chart_type == 'bar':
            if y_column:
                sns.barplot(data=df, x=x_column, y=y_column, hue=hue_column, errorbar=None)
            else:
                sns.countplot(data=df, x=x_column, hue=hue_column)
        elif chart_type == 'pie':
            counts = df[x_column].value_counts()
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        elif chart_type == 'box':
            sns.boxplot(data=df, x=x_column, y=y_column, hue=hue_column)
        elif chart_type == 'violin':
            sns.violinplot(data=df, x=x_column, y=y_column, hue=hue_column, split=True if hue_column and df[hue_column].nunique() == 2 else False)
        elif chart_type == 'scatterplot':
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
        elif chart_type == 'heatmap':
            # For heatmap, x_column and y_column can act as dimensions, or if none provided, do correlation heatmap for numerics
            if x_column and y_column and hue_column:
                # e.g., pivot table
                pivot = df.pivot_table(index=y_column, columns=x_column, values=hue_column, aggfunc='mean')
                sns.heatmap(pivot, annot=True, cmap="coolwarm")
            else:
                numeric_df = df.select_dtypes(include=['number'])
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        else:
            return f"Error: Unsupported chart type '{chart_type}'."
            
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(f"{chart_type.capitalize()} of {x_column}" + (f" vs {y_column}" if y_column else ""), fontsize=16)
            
        plt.tight_layout()
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(CHART_DIR, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        return f"Chart successfully generated and saved at: {filepath}"
        
    except Exception as e:
        return f"Error generating visualization: {str(e)}"
