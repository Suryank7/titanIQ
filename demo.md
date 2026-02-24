# Demo Queries

Here are some test questions you can throw at the agent to test its tool routing and context retrieval.

## Data Exploration
Tests the schema and pandas tools:
- "How many passengers are in the dataset?"
- "What are the columns and their data types in this dataset?"
- "How many passengers survived versus died?"
- "What was the average fare paid by passengers?"

## Statistical Analysis
Tests the pandas and stats tools for math and aggregations:
- "What was the survival rate of 1st class passengers compared to 3rd class?"
- "Did women have a higher survival rate than men?"
- "What is the correlation between the fare a passenger paid and their age?"
- "What was the maximum age of someone who survived?"

## Visualizations
Tests the visualization tool's capacity to generate specific seaborn charts:
- "Plot a pie chart showing the percentage of survivors vs non-survivors."
- "Generate a bar chart of passenger counts grouped by Ticket Class."
- "Show me a histogram of passenger ages."
- "Visualize the survival rate by gender using a bar chart."
- "Generate a correlation heatmap for numerical features."

## Context & Memory
Ask these back-to-back to test LangGraph's conversational history state:
- "What was the survival rate by gender?"
- "Now show me that same breakdown but only for 1st class passengers."
- "Can you visualize that last insight for me?"

## ML Predictions
Tests the survival predictor tool using the pre-trained Scikit-Learn RF model:
- "What are the survival chances of a 25-year-old male traveling in 3rd class who paid a $10 fare?"
- "If I am a 40-year-old female in 1st class from Cherbourg, what are my chances of survival?"

## Unstructured Exploration
Tests the agent's ability to chain logical tools together without a strict directive:
- "Tell me something interesting about the data."
- "Who were the most likely people to survive the Titanic sink?"
- "What factor played the biggest role in a passenger surviving?"
