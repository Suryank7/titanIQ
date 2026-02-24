# TitanIQ: AI Data Analyst

An experimental autonomous agent built to analyze the Titanic dataset. It uses LangGraph and Groq (Llama-3.3-70B) to parse natural language queries, execute Pandas data operations, calculate stats, and generate Seaborn charts on the fly. 

The project also includes a lightweight Random Forest model for running "what-if" survival predictions directly from the UI.

## Stack
- **Agent**: LangGraph, langchain-groq (Llama-3.3-70B)
- **API**: FastAPI
- **UI**: Streamlit
- **Data**: Pandas, Scikit-learn, Matplotlib, Seaborn

## Setup & Deployment

You can run this project on Streamlit Cloud, Docker, or directly on your local machine.

### 1. Streamlit Community Cloud (Easiest)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app pointing to your repo.
3. Set the **Main file path** to `frontend/app.py`.
4. In Advanced Settings > Secrets, add your API key:
   ```toml
   GROQ_API_KEY="gsk_your_key_here"
   ```
5. Deploy.

### 2. Docker
1. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
2. Run `docker-compose up --build`.
3. Open http://localhost:8501.
4. FastAPI docs are at http://localhost:8000/docs.

### 3. Local Environment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set your `GROQ_API_KEY`.
3. Pre-train the ML survival model (needed for the simulator sidebar):
   ```bash
   python backend/models/train_model.py
   ```
4. Start the frontend:
   ```bash
   python -m streamlit run frontend/app.py
   ```

## Example Prompts
- "What's the correlation between Fare and Age?"
- "Plot the survival count grouped by Ticket Class."
- "Show me a histogram of passenger ages, separated by survival status."
- "What are the survival chances of a 25-year-old male traveling in 3rd class?"
