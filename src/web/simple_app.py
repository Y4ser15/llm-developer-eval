# src/web/simple_app.py
"""
SIMPLE Web Interface for LLM Coding Evaluation Platform - NO PROGRESS CALLBACKS
Focus on working evaluation without progress complexity.
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Form, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
import os
import json
from datetime import datetime
from pathlib import Path

# Import our evaluation components - SIMPLE VERSION
from ..core.model_interfaces import ModelConfig, ModelFactory
from ..evaluation.simple_evaluator import ComprehensiveEvaluator, EvaluationConfig
from ..core.simple_datasets import SimpleBenchmarkOrchestrator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Coding Evaluation Platform",
    description="Simple evaluation platform for LLMs",
    version="3.0.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

# Initialize components - SIMPLE VERSION
evaluator = ComprehensiveEvaluator()
orchestrator = SimpleBenchmarkOrchestrator()

# Background task storage
active_evaluations: Dict[str, Dict] = {}

# Simple WebSocket connections (no progress updates, just connection)
active_connections: Dict[str, WebSocket] = {}


# Simple WebSocket for compatibility (no progress updates)
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Simple WebSocket endpoint - just maintains connection"""
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        # Just keep connection alive, no progress updates
        while client_id in active_evaluations:
            await asyncio.sleep(1)
            # Send simple keep-alive
            if active_evaluations[client_id]["status"] == "completed":
                await websocket.send_text(json.dumps({
                    "type": "progress", 
                    "message": f"🎉 Evaluation completed! Run ID: {active_evaluations[client_id].get('run_id', 'unknown')}", 
                    "progress": 100
                }))
                break
    except:
        pass
    finally:
        if client_id in active_connections:
            del active_connections[client_id]


# Pydantic models for API
class ModelConfigRequest(BaseModel):
    name: str
    provider: str
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096


class SimpleEvaluationRequest(BaseModel):
    model_configs: List[ModelConfigRequest]
    domains: List[str] = ["frontend", "backend", "testing"]
    max_tasks_per_domain: int = 5
    include_humaneval: bool = True
    generate_report: bool = True


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    
    # Get recent evaluation results
    recent_results = evaluator.list_evaluation_results()[:5]
    
    # Get available models count
    available_models = _get_available_models()
    
    # Get benchmark status
    humaneval_available = orchestrator.humaneval.available
    
    return templates.TemplateResponse(
        "comprehensive_home.html",
        {
            "request": request,
            "title": "LLM Coding Evaluation Platform",
            "recent_results": recent_results,
            "available_models": len(available_models),
            "bigcodebench_available": False,  # We don't use this
            "humaneval_available": humaneval_available,
            "total_evaluations": len(recent_results)
        }
    )


@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    """Evaluation configuration page"""
    
    # Get available models
    available_models = _get_available_models()
    
    return templates.TemplateResponse(
        "comprehensive_evaluate.html",
        {
            "request": request,
            "title": "Start Evaluation",
            "available_models": available_models,
            "supported_domains": ["frontend", "backend", "testing"]
        }
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Results dashboard"""
    
    # Get all evaluation results
    try:
        all_results = evaluator.list_evaluation_results()
    except Exception as e:
        logger.error(f"Failed to list evaluation results: {e}")
        all_results = []
    
    # Get latest leaderboard
    leaderboard = []
    if all_results:
        latest_result_file = all_results[0]["file_path"]
        try:
            with open(latest_result_file, 'r') as f:
                data = json.load(f)
                leaderboard = data.get("leaderboard", [])
                
                # If no leaderboard in the data, create one from models
                if not leaderboard and "models" in data:
                    leaderboard = []
                    for i, model in enumerate(data["models"]):
                        summary = model.get("summary", {})
                        leaderboard.append({
                            "rank": i + 1,
                            "model_name": model.get("model_name", "Unknown"),
                            "provider": model.get("provider", "Unknown"),
                            "overall_score": summary.get("overall_score", 0),
                            "pass_rate": summary.get("passed_tasks", 0) / max(summary.get("total_tasks", 1), 1),
                            "domain_scores": summary.get("domain_scores", {}),
                            "execution_time": summary.get("execution_time", 0)
                        })
        except Exception as e:
            logger.error(f"Failed to load latest results: {e}")
            leaderboard = []
    
    # Simple benchmark statistics
    benchmark_stats = {
        "total_evaluations": len(all_results),
        "total_models_evaluated": sum(result.get("model_count", 0) for result in all_results),
        "average_duration": sum(result.get("duration", 0) for result in all_results) / max(len(all_results), 1),
        "domain_performance": {
            "frontend": {"average": 0, "count": 0},
            "backend": {"average": 0, "count": 0},
            "testing": {"average": 0, "count": 0}
        }
    }
    
    return templates.TemplateResponse(
        "comprehensive_dashboard.html",
        {
            "request": request,
            "title": "Results Dashboard",
            "leaderboard": leaderboard,
            "all_results": all_results,
            "benchmark_stats": benchmark_stats,
            "active_evaluations": len(active_evaluations)
        }
    )


@app.get("/results/{run_id}")
async def get_evaluation_result(request: Request, run_id: str):
    """Get specific evaluation result"""
    
    try:
        result_file = evaluator.results_dir / f"comprehensive_results_{run_id}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            return templates.TemplateResponse(
                "evaluation_result.html",
                {
                    "request": request,
                    "title": f"Evaluation Results - {run_id}",
                    "result_data": data,
                    "run_id": run_id
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Evaluation result not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{run_id}/report")
async def get_evaluation_report(run_id: str):
    """Get HTML report for evaluation"""
    
    report_file = evaluator.results_dir / f"comprehensive_report_{run_id}.html"
    if report_file.exists():
        return FileResponse(report_file, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Report not found")


@app.post("/api/evaluate")
async def start_evaluation(
    evaluation_request: SimpleEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Start evaluation in background - NO PROGRESS CALLBACKS"""
    
    try:
        # Convert request to internal format
        model_configs = []
        for model_req in evaluation_request.model_configs:
            config = ModelConfig(**model_req.dict())
            model_configs.append(config)
        
        eval_config = EvaluationConfig(
            domains=evaluation_request.domains,
            max_tasks_per_domain=evaluation_request.max_tasks_per_domain,
            include_humaneval=evaluation_request.include_humaneval,
            generate_report=evaluation_request.generate_report,
            save_results=True
        )
        
        # Generate client ID
        client_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start evaluation in background
        background_tasks.add_task(
            run_simple_evaluation_task,
            model_configs,
            eval_config,
            client_id
        )
        
        # Track active evaluation
        active_evaluations[client_id] = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "models": [config.name for config in model_configs],
            "domains": evaluation_request.domains
        }
        
        return {
            "status": "started",
            "message": "Evaluation started successfully",
            "client_id": client_id,
            "estimated_duration": len(model_configs) * len(evaluation_request.domains) * evaluation_request.max_tasks_per_domain * 2  # 2 seconds per task estimate
        }
        
    except Exception as e:
        logger.error(f"Failed to start evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_simple_evaluation_task(
    model_configs: List[ModelConfig],
    eval_config: EvaluationConfig,
    client_id: str
):
    """Run evaluation task in background - NO PROGRESS CALLBACKS"""
    
    try:
        logger.info(f"🚀 Starting evaluation task {client_id}")
        
        # Update status
        active_evaluations[client_id]["status"] = "running"
        
        # Run evaluation (no progress callbacks)
        evaluation_run = await evaluator.evaluate_models(model_configs, eval_config)
        
        # Update status
        active_evaluations[client_id]["status"] = "completed"
        active_evaluations[client_id]["run_id"] = evaluation_run.run_id
        active_evaluations[client_id]["duration"] = evaluation_run.total_duration
        
        logger.info(f"✅ Evaluation task {client_id} completed: {evaluation_run.run_id}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation task {client_id} failed: {e}")
        active_evaluations[client_id]["status"] = "failed"
        active_evaluations[client_id]["error"] = str(e)
    
    finally:
        # Clean up after delay
        await asyncio.sleep(300)  # Keep status for 5 minutes
        if client_id in active_evaluations:
            del active_evaluations[client_id]


@app.get("/api/models/discover")
async def discover_models():
    """Discover actually available models"""
    discovered = []
    
    # Check Ollama models
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_models = response.json().get("models", [])
            for model in ollama_models:
                discovered.append({
                    "name": f"Ollama - {model['name']}",
                    "provider": "ollama",
                    "model_name": model['name'],
                    "base_url": "http://localhost:11434",
                    "available": True,
                    "discovered": True
                })
    except Exception as e:
        logger.debug(f"Ollama not available: {e}")
    
    return {"discovered_models": discovered}


@app.get("/api/models/available")
async def get_available_models():
    """Get list of available models"""
    return _get_available_models()


@app.post("/api/huggingface/login")
async def huggingface_login(request: Request):
    """Handle HuggingFace authentication - SIMPLIFIED"""
    try:
        form_data = await request.form()
        token = form_data.get("token")
        
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        
        return {
            "status": "success",
            "message": "Token accepted (not needed for HumanEval)",
            "user": "user"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/huggingface/status")
async def huggingface_status():
    """Check HuggingFace authentication status - SIMPLIFIED"""
    return {
        "authenticated": True,  # Always true since we don't need HF for HumanEval
        "user": "user"
    }


@app.get("/api/benchmarks/status")
async def get_benchmark_status():
    """Get status of available benchmarks"""
    datasets_status = await orchestrator.dataset_manager.get_available_datasets()
    return {
        "humaneval": {
            "available": datasets_status.get('humaneval', False),
            "description": "164 Python programming problems (streaming)",
            "domains": ["general"]
        }
    }


@app.get("/api/evaluations/status/{client_id}")
async def get_evaluation_status(client_id: str):
    """Get evaluation status by client ID"""
    if client_id in active_evaluations:
        return active_evaluations[client_id]
    else:
        raise HTTPException(status_code=404, detail="Evaluation not found")


@app.get("/api/evaluations/active")
async def get_active_evaluations():
    """Get list of active evaluations"""
    return active_evaluations


@app.get("/api/results")
async def get_all_results():
    """Get list of all evaluation results"""
    return evaluator.list_evaluation_results()


# Utility functions
def _get_available_models() -> List[Dict[str, Any]]:
    """Get list of available model configurations"""
    
    models = []
    
    # Add API models (check env vars)
    api_models = [
        {
            "name": "GPT-4 Turbo",
            "provider": "openai",
            "model_name": "gpt-4-turbo-preview",
            "available": bool(os.getenv("OPENAI_API_KEY"))
        },
        {
            "name": "GPT-4o",
            "provider": "openai",
            "model_name": "gpt-4o",
            "available": bool(os.getenv("OPENAI_API_KEY"))
        },
        {
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "model_name": "claude-3-5-sonnet-20241022",
            "available": bool(os.getenv("ANTHROPIC_API_KEY"))
        }
    ]
    
    models.extend(api_models)
    
    # Add configurable servers
    models.extend([
        {
            "name": "vLLM Server (Configure)",
            "provider": "vllm",
            "model_name": "configure-vllm",
            "base_url": "http://localhost:8000",
            "available": True,
            "configurable": True,
            "description": "Add your vLLM server endpoint"
        },
        {
            "name": "Custom Server (Configure)",
            "provider": "custom",
            "model_name": "configure-custom",
            "base_url": "http://localhost:8000",
            "available": True,
            "configurable": True,
            "description": "Add your custom LLM API endpoint"
        }
    ])
    
    return models


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "humaneval_available": orchestrator.humaneval.available,
        "active_evaluations": len(active_evaluations)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
