from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import csv
import io
import asyncio
from datetime import datetime
from openai import AsyncOpenAI

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
    model_name: str = "gpt-4.1"

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
            
            is_correct, reasoning = await _evaluate_correctness(completion, ground_truth)
            
            case_results["predictions"][model_key] = {
                "completion": completion,
                "is_correct": is_correct,
                "reasoning": reasoning,
                "latency": latency,
                "tool_calls": len(tool_calls),
                "trace": tool_calls  # Store the full trace
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

async def _evaluate_correctness(prediction: Optional[str], ground_truth: str) -> tuple[bool, str]:
    if not prediction and ground_truth != "NO_COMPLETION_NEEDED":
        return False, "No prediction provided"
    
    # Use GPT-4.1 as judge
    try:
        client = AsyncOpenAI()
        
        if ground_truth == "NO_COMPLETION_NEEDED":
            prompt = f"""You are evaluating whether a model correctly identified that no completion was needed.

The model's response: {prediction if prediction else "[empty/no response]"}

The model should have either:
1. Returned nothing/empty string
2. Indicated that no completion is needed

Did the model correctly avoid providing a completion?

Please respond with your judgment in the following XML format:
<judgment>
<verdict>YES/NO</verdict>
<reasoning>Your explanation for why you made this decision</reasoning>
</judgment>"""
        else:
            prompt = f"""You are evaluating whether a model's prediction matches the ground truth for a financial data AUTOCOMPLETE task.

Ground Truth (the completion only): {ground_truth}
Model Prediction: {prediction}

CRITICAL: This is an autocomplete task, so the model should return ONLY the completion text, NOT the full sentence.
For example, if the input is "Apple's revenue in 2023 was", the model should return "$383.3 billion", NOT "Apple's revenue in 2023 was $383.3 billion".

Please determine if the model's prediction is correct. Consider:
- The model MUST return only the completion suffix, not repeat the input
- Numeric values should be approximately equal (within reasonable rounding)
- Different formats are acceptable (e.g., "$1.2B" vs "$1.2 billion" vs "1200 million")
- Formatting differences are acceptable
- We care about the meaning, not the exact symbols or how natural the language is
- 0 does not mean no completion needed

Please respond with your judgment in the following XML format:
<judgment>
<verdict>YES/NO</verdict>
<reasoning>Your explanation for why you made this decision</reasoning>
</judgment>"""

        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse XML response
        import re
        verdict_match = re.search(r'<verdict>(YES|NO)</verdict>', response_text, re.IGNORECASE)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL)
        
        if verdict_match:
            verdict = verdict_match.group(1).upper() == "YES"
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            return verdict, reasoning
        else:
            # Fallback if XML parsing fails
            return "YES" in response_text.upper(), f"Failed to parse XML response: {response_text}"
        
    except Exception as e:
        print(f"Error using GPT-4.1 judge: {e}")
        # Fallback to simple exact match
        is_correct = prediction.lower().strip() == ground_truth.lower().strip()
        return is_correct, f"Fallback evaluation due to error: {str(e)}"

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