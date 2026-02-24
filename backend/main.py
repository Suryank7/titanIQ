import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re

from backend.agent.llm import chat_with_agent
from backend.tools.survival_predictor_tool import predict_survival

app = FastAPI(
    title="Titanic Autonomous AI Data Analyst API",
    description="Production-Grade Autonomous AI Data Analyst Agent for Titanic Dataset",
    version="1.0.0"
)

# Serve generated charts statically
os.makedirs("frontend/charts", exist_ok=True)
app.mount("/charts", StaticFiles(directory="frontend/charts"), name="charts")

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Titanic Autonomous AI Data Analyst API"}

# Chat Endpoint Models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    visualizations: List[str] = []
    data_payload: Optional[Dict[str, Any]] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = chat_with_agent(request.query, request.session_id)
        
        # Extract image paths if the agent mentions successfully generated and saved at: frontend/charts/...
        # The agent returns something like: Chart successfully generated and saved at: frontend/charts/abc.png
        # We can regex it or just return the text and let frontend parse. 
        # But let's proactively parse it into visuals list
        visualizations = []
        import re
        matches = re.findall(r'frontend[\\/]charts[\\/]([a-f0-9-]+\.png)', response_text)
        for filename in matches:
            visualizations.append(f"/charts/{filename}")
            
        return ChatResponse(
            response=response_text,
            visualizations=visualizations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AnalyzeRequest(BaseModel):
    sql_or_pandas_query: str

@app.post("/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    # TODO: Implement direct data queries safely
    return {"status": "success", "result": "Stub result"}

class VisualizeRequest(BaseModel):
    chart_type: str
    x_axis: str
    y_axis: Optional[str] = None
    group_by: Optional[str] = None
    title: Optional[str] = None

@app.post("/visualize")
async def visualize_endpoint(request: VisualizeRequest):
    # TODO: Implement dynamic chart generation
    return {"status": "success", "image_path": "stub_path.png"}

class PredictRequest(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str

@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    try:
        # direct tool call via predict_survival.invoke since LangChain tools wrap the func
        result = predict_survival.invoke({
            "pclass": request.Pclass,
            "sex": request.Sex,
            "age": request.Age,
            "fare": request.Fare,
            "embarked": request.Embarked
        })
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def schema_endpoint():
    # TODO: Return dataset metadata
    return {
        "dataset": "Titanic",
        "description": "Information about passengers on the Titanic.",
        "columns": [
            "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
        ]
    }
