# 🚢 Titanic Autonomous AI Data Analyst - Demo Questions

Here is a list of curated demo questions you can use to showcase the capabilities of your AI Data Analyst Agent.

These questions test the agent's ability to query data, calculate statistics, generate visuals, and retain conversational context.

## 📊 1. Basic Data Exploration
*These questions show how well the AI queries and understands the dataset schema.*
- "How many passengers are in the dataset?"
- "What are the columns and their data types in this dataset?"
- "How many passengers survived versus died?"
- "What was the average fare paid by passengers?"

## 📈 2. Statistical Analysis & Comparisons
*These questions trigger the `pandas_tool` and `stats_tool` to perform mathematical aggregations.*
- "What was the survival rate of 1st class passengers compared to 3rd class?"
- "Did women have a higher survival rate than men?"
- "What is the correlation between the fare a passenger paid and their age?"
- "What was the maximum age of someone who survived?"

## 🎨 3. Visualizations (Chart Generation)
*These prompts trigger the `visualization_tool` to generate native UI charts.*
- "Plot a pie chart showing the percentage of survivors vs non-survivors."
- "Generate a bar chart of passenger counts grouped by Ticket Class."
- "Show me a histogram of passenger ages."
- "Visualize the survival rate by gender using a bar chart."
- "Generate a correlation heatmap for numerical features."

## 🧩 4. Context & Memory (Follow-ups)
*Ask these back-to-back to demonstrate conversational memory mechanics.*
- **User:** "What was the survival rate by gender?"
- **User:** "Now show me that same breakdown but only for 1st class passengers."
- **User:** "Can you visualize that last insight for me?"

## 🧮 5. "What-if" ML Predictions
*These questions show off the `survival_predictor_tool` hitting the trained Scikit-Learn RandomForest model.*
- "What are the survival chances of a 25-year-old male traveling in 3rd class who paid a $10 fare?"
- "If I am a 40-year-old female in 1st class from Cherbourg, what are my chances of survival?"

## ✨ 6. Autonomous / Vague Exploration
*Tests the AI's ability to "act like an analyst" without explicit instructions.*
- "Tell me something interesting about the data."
- "Who were the most likely people to survive the Titanic sink?"
- "What factor played the biggest role in a passenger surviving?"
