from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import csv
import io
import asyncio
from datetime import datetime

import sys
import os

# Add parent directory to path if running as a script
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)

try:
    from backend.database import init_database, get_db, get_all_tickers, get_all_metrics
    from backend.agent import AgentExecutor
except ImportError:
    from database import init_database, get_db, get_all_tickers, get_all_metrics
    from agent import AgentExecutor

# This will be replaced after lifespan definition

class CompletionRequest(BaseModel):
    text: str
    model_provider: str = "openai"
    model_name: str = "gpt-4o-mini"

class CompletionResponse(BaseModel):
    completion: Optional[str]
    tool_calls: List[Dict[str, Any]]
    latency: float
    model: str

class BatchEvaluationRequest(BaseModel):
    test_cases: List[Dict[str, str]]
    models: List[Dict[str, str]]

class BatchEvaluationResponse(BaseModel):
    results: List[Dict[str, Any]]
    accuracy_scores: Dict[str, float]

class FinancialDataRequest(BaseModel):
    ticker: Optional[str] = None
    metric: Optional[str] = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    yield
    # Shutdown (if needed)

app = FastAPI(title="Financial AI Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/completion", response_model=CompletionResponse)
async def get_completion(request: CompletionRequest):
    executor = AgentExecutor(
        model_provider=request.model_provider,
        model_name=request.model_name
    )
    
    completion, tool_calls, latency = await executor.get_completion(request.text)
    
    return CompletionResponse(
        completion=completion,
        tool_calls=tool_calls,
        latency=latency,
        model=f"{request.model_provider}/{request.model_name}"
    )

@app.post("/api/batch-evaluation", response_model=BatchEvaluationResponse)
async def batch_evaluation(request: BatchEvaluationRequest):
    results = []
    model_correct_counts = {f"{m['provider']}/{m['name']}": 0 for m in request.models}
    
    for test_case in request.test_cases:
        input_text = test_case.get("input", "")
        ground_truth = test_case.get("ground_truth", "")
        
        case_results = {
            "input": input_text,
            "ground_truth": ground_truth,
            "predictions": {}
        }
        
        for model in request.models:
            executor = AgentExecutor(
                model_provider=model["provider"],
                model_name=model["name"]
            )
            
            completion, tool_calls, latency = await executor.get_completion(input_text)
            
            model_key = f"{model['provider']}/{model['name']}"
            
            is_correct = await _evaluate_correctness(completion, ground_truth)
            
            case_results["predictions"][model_key] = {
                "completion": completion,
                "is_correct": is_correct,
                "latency": latency,
                "tool_calls": len(tool_calls)
            }
            
            if is_correct:
                model_correct_counts[model_key] += 1
        
        results.append(case_results)
    
    accuracy_scores = {
        model: correct / len(request.test_cases) if request.test_cases else 0
        for model, correct in model_correct_counts.items()
    }
    
    return BatchEvaluationResponse(
        results=results,
        accuracy_scores=accuracy_scores
    )

async def _evaluate_correctness(prediction: Optional[str], ground_truth: str) -> bool:
    if not prediction:
        return False
    
    prediction_normalized = prediction.lower().strip()
    ground_truth_normalized = ground_truth.lower().strip()
    
    if prediction_normalized == ground_truth_normalized:
        return True
    
    try:
        pred_nums = [float(s) for s in prediction.replace(",", "").split() if s.replace(".", "").replace("-", "").isdigit()]
        truth_nums = [float(s) for s in ground_truth.replace(",", "").split() if s.replace(".", "").replace("-", "").isdigit()]
        
        if pred_nums and truth_nums:
            for pred in pred_nums:
                for truth in truth_nums:
                    if abs(pred - truth) / max(abs(truth), 1) < 0.05:
                        return True
    except:
        pass
    
    return False

@app.post("/api/upload-evaluation-csv")
async def upload_evaluation_csv(file: UploadFile = File(...)):
    content = await file.read()
    csv_reader = csv.DictReader(io.StringIO(content.decode()))
    
    test_cases = []
    for row in csv_reader:
        test_cases.append({
            "input": row.get("input", ""),
            "ground_truth": row.get("ground_truth", "")
        })
    
    return {"test_cases": test_cases}

@app.get("/api/financial-data")
async def get_financial_data(ticker: Optional[str] = None):
    async with get_db() as db:
        if ticker:
            query = """
                SELECT fd.*, m.description, m.unit as metric_unit
                FROM financial_data fd
                JOIN metrics m ON fd.metric_name = m.metric_name
                WHERE fd.ticker = ?
                ORDER BY fd.period DESC, fd.metric_name
            """
            async with db.execute(query, (ticker,)) as cursor:
                rows = await cursor.fetchall()
                return {"data": [dict(row) for row in rows]}
        else:
            query = """
                SELECT DISTINCT ticker, COUNT(DISTINCT metric_name) as metric_count,
                       COUNT(DISTINCT period) as period_count
                FROM financial_data
                GROUP BY ticker
                ORDER BY ticker
            """
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                return {"tickers": [dict(row) for row in rows]}

@app.get("/api/tickers")
async def get_tickers():
    tickers = await get_all_tickers()
    return {"tickers": tickers}

@app.get("/api/metrics")
async def get_metrics():
    metrics = await get_all_metrics()
    return {"metrics": metrics}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)