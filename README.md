# Titanic Autonomous AI Data Analyst

Welcome to the **Titanic Autonomous AI Data Analyst** application! This project provides a conversational AI agent capable of intelligent data analysis, visualization generation, and survival prediction on the Titanic dataset, powered by the cutting-edge **Groq Llama-3.3-70B** through a LangGraph agent loop.

## 🚀 Features

- **Natural Language Data Queries**: Ask "What is the survival rate by gender?" and get a direct answer based on real data.
- **Dynamic Visualizations**: The agent can generate Histograms, Scatterplots, Bar Charts, Heatmaps, Box, and Violin plots and output them directly to the UI.
- **Conversational Memory**: The agent remembers previous queries, so you can do follow-ups.
- **Survival Simulator**: A trained Random Forest ML model predicts the chances of survival for any given passenger profile immediately.
- **Schema & Correlation Explorers**: The agent has native tools to understand dataset dimensions and analyze feature correlations instantly.

## 🏗️ Technical Stack
- **AI Brain**: `langgraph`, `langchain-groq` (Groq Llama-3.3-70B Versatile).
- **Backend API**: `FastAPI` (Async REST architecture).
- **Frontend UI**: `Streamlit`.
- **Data & ML**: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

## 📦 Setup & Deployment

There are multiple ways to deploy this system:

### Option 1: Streamlit Community Cloud (Recommended)
This repository is optimized for immediate deployment to Streamlit Cloud!
1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a **New app** from the repository.
3. Set the **Main file path** to `frontend/app.py`.
4. In the **Advanced Settings > Secrets**, paste your API key:
```toml
GROQ_API_KEY="gsk_your_key_here"
```
5. Click **Deploy**.

### Option 2: Docker (Local Containers)
1. Provide a Groq API key: Copy `.env.example` to `.env` and fill in `GROQ_API_KEY=gsk_...`.
2. Run Docker Compose:
```bash
docker-compose up --build
```
3. Visit the app: http://localhost:8501
4. Fast API backend Swagger Docs: http://localhost:8000/docs

### Option 3: Local Python Environment
1. Setup a virtual environment (optional) and install dependencies:
```bash
pip install -r requirements.txt
```
2. Copy `.env.example` to `.env` and configure `GROQ_API_KEY=gsk_...`.
3. Train the ML survival model (must run once before using simulator):
```bash
python backend/models/train_model.py
```
4. Start the Application:
```bash
python -m streamlit run frontend/app.py
```

## 🧠 Example Queries
- *What is the correlation between Fare and Age in the dataset?*
- *Can you visualize the survival count grouped by Ticket Class?*
- *Tell me something interesting about the data.*
- *Generate a histogram of passenger ages, separated by survival status.*
