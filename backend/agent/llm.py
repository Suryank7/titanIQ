import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import tools
from backend.tools.pandas_tool import execute_pandas_query
from backend.tools.stats_tool import calculate_statistics
from backend.tools.visualization_tool import generate_visualization
from backend.tools.schema_tool import get_dataset_schema
from backend.tools.correlation_tool import analyze_correlation
from backend.tools.survival_predictor_tool import predict_survival

# Simple in-memory session store for conversational history
session_store = {}

# Ensure the API key is accessible
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("WARNING: GROQ_API_KEY not found in environment variables. Agent will fail if not provided.")

system_prompt = """You are a Senior Data Analyst AI expert answering questions about the Titanic dataset.
You have access to a suite of tools. 
When asked a question, use your tools to analyze the data, generate visualizations if requested or helpful, and provide insights.
If the user asks a vague question like "tell me something interesting", explore the dataset to find significant patterns (e.g., survival by gender or class) and optionally generate a chart to explain it.

Rules:
1. Always be conversational, clear, and professional.
2. Explain the results in plain English, providing context (e.g., "This indicates...").
3. Do NOT invent data. Use your tools to find the exact numbers. 
4. If a chart is generated, its file path will be returned by the tool. If the user didn't explicitly ask for a chart but it heavily aids your explanation, feel free to generate one anyway.
5. If the user asks about the survival chances of a specific person profile, use the `predict_survival` tool.
"""

def get_agent_executor():
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=api_key)
    
    tools = [
        execute_pandas_query,
        calculate_statistics,
        generate_visualization,
        get_dataset_schema,
        analyze_correlation,
        predict_survival
    ]
    
    # LangGraph create_react_agent
    agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
    return agent_executor

def chat_with_agent(query: str, session_id: str = "default"):
    # Initialize session history if not present
    if session_id not in session_store:
        session_store[session_id] = []
        
    chat_history = session_store[session_id]
    
    try:
        executor = get_agent_executor()
        
        # Format history for LangGraph
        messages = chat_history + [HumanMessage(content=query)]
        
        result = executor.invoke({"messages": messages})
        
        # 'messages' key contains the full conversation turn
        final_messages = result.get('messages', [])
        response_text = final_messages[-1].content if final_messages else "I'm sorry, I couldn't process that."
        
        # Look for charts in ALL messages in this turn
        visualizations = []
        import re
        for msg in final_messages:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                matches = re.findall(r'frontend[\\/]charts[\\/]([a-f0-9-]+\.png)', content)
                for m in matches:
                    vis_path = os.path.join("frontend", "charts", m)
                    if vis_path not in visualizations:
                        visualizations.append(vis_path)
        
        # Save to memory (limit history to last 10 turns to save tokens)
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response_text))
        session_store[session_id] = chat_history[-10:]
        
        return {"text": response_text, "visualizations": visualizations}
    except Exception as e:
        return {"text": f"Agent Error: {str(e)}", "visualizations": []}
