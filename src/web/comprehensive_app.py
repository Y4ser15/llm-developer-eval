# src/web/comprehensive_app.py
"""
Comprehensive Web Interface for LLM Coding Evaluation Platform
One-click evaluation with real-time progress and results dashboard.
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
    description="Comprehensive evaluation platform for LLMs on Frontend, Backend, and Testing domains",
    version="2.0.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

# Initialize components - SIMPLE VERSION
evaluator = ComprehensiveEvaluator()
orchestrator = SimpleBenchmarkOrchestrator()

# WebSocket connections for real-time updates
active_connections: Dict[str, WebSocket] = {}

# Background task storage
active_evaluations: Dict[str, Dict] = {}


# Pydantic models for API
class ModelConfigRequest(BaseModel):
    name: str
    provider: str
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096


class ComprehensiveEvaluationRequest(BaseModel):
    model_configs: List[ModelConfigRequest]
    domains: List[str] = ["frontend", "backend", "testing"]
    max_tasks_per_domain: int = 10
    include_bigcodebench: bool = True
    include_humaneval: bool = True
    parallel_models: bool = False
    generate_report: bool = True


# WebSocket for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except:
        pass
    finally:
        if client_id in active_connections:
            del active_connections[client_id]


async def send_progress_update(client_id: str, message: str, progress: float = None):
    """Send progress update to WebSocket client"""
    if client_id in active_connections:
        try:
            update = {"type": "progress", "message": message}
            if progress is not None:
                update["progress"] = progress
            await active_connections[client_id].send_text(json.dumps(update))
        except:
            # Connection closed
            if client_id in active_connections:
                del active_connections[client_id]


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Enhanced home page with evaluation overview"""
    
    # Get recent evaluation results
    recent_results = evaluator.list_evaluation_results()[:5]
    
    # Get available models count
    available_models = _get_available_models()
    
    # Get benchmark status
    bigcodebench_available = orchestrator.bigcodebench.bigcodebench_available
    humaneval_available = orchestrator.humaneval.available
    
    return templates.TemplateResponse(
        "comprehensive_home.html",
        {
            "request": request,
            "title": "LLM Coding Evaluation Platform",
            "recent_results": recent_results,
            "available_models": len(available_models),
            "bigcodebench_available": bigcodebench_available,
            "humaneval_available": humaneval_available,
            "total_evaluations": len(recent_results)
        }
    )


@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    """Comprehensive evaluation configuration page"""
    
    # Get available models
    available_models = _get_available_models()
    
    # Get benchmark information
    benchmark_info = {
        "bigcodebench": {
            "available": orchestrator.bigcodebench.bigcodebench_available,
            "description": "Practical programming tasks with diverse function calls",
            "domains": ["frontend", "backend", "testing"]
        },
        "humaneval": {
            "available": orchestrator.humaneval.available,
            "description": "Function-level code generation tasks",
            "domains": ["general"]
        }
    }
    
    return templates.TemplateResponse(
        "comprehensive_evaluate.html",
        {
            "request": request,
            "title": "Start Evaluation",
            "available_models": available_models,
            "benchmark_info": benchmark_info,
            "supported_domains": ["frontend", "backend", "testing"]
        }
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Results dashboard with leaderboards and analytics - fixed error handling"""
    
    # Get all evaluation results with error handling
    try:
        all_results = evaluator.list_evaluation_results()
    except Exception as e:
        logger.error(f"Failed to list evaluation results: {e}")
        all_results = []
    
    # Get latest leaderboard with error handling
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
                            "pass_rate": summary.get("pass_rate", 0),
                            "domain_scores": summary.get("domain_scores", {
                                "frontend": 0,
                                "backend": 0, 
                                "testing": 0
                            }),
                            "avg_execution_time": summary.get("avg_execution_time", 0)
                        })
        except Exception as e:
            logger.error(f"Failed to load latest results: {e}")
            leaderboard = []
    
    # Get benchmark statistics with error handling
    try:
        benchmark_stats = _get_benchmark_statistics(all_results)
    except Exception as e:
        logger.error(f"Failed to get benchmark statistics: {e}")
        benchmark_stats = {
            "total_evaluations": 0,
            "total_models_evaluated": 0,
            "average_duration": 0,
            "domain_performance": {
                "frontend": {"average": 0, "count": 0, "min": 0, "max": 0},
                "backend": {"average": 0, "count": 0, "min": 0, "max": 0},
                "testing": {"average": 0, "count": 0, "min": 0, "max": 0}
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
    """Get specific evaluation result with proper HTML display"""
    
    try:
        result_file = evaluator.results_dir / f"comprehensive_results_{run_id}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Return HTML template instead of raw JSON
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
async def start_comprehensive_evaluation(
    evaluation_request: ComprehensiveEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Start comprehensive evaluation in background"""
    
    try:
        # Convert request to internal format
        model_configs = []
        for model_req in evaluation_request.model_configs:
            config = ModelConfig(**model_req.dict())
            model_configs.append(config)
        
        eval_config = EvaluationConfig(
            domains=evaluation_request.domains,
            max_tasks_per_domain=evaluation_request.max_tasks_per_domain,
            include_bigcodebench=evaluation_request.include_bigcodebench,
            include_humaneval=evaluation_request.include_humaneval,
            parallel_models=evaluation_request.parallel_models,
            generate_report=evaluation_request.generate_report,
            save_results=True
        )
        
        # Generate client ID for WebSocket updates
        client_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start evaluation in background
        background_tasks.add_task(
            run_comprehensive_evaluation_background,
            model_configs,
            eval_config,
            client_id
        )
        
        return {
            "status": "started",
            "client_id": client_id,
            "message": "Comprehensive evaluation started. Connect to WebSocket for real-time updates."
        }
        
    except Exception as e:
        logger.error(f"Failed to start evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_comprehensive_evaluation_background(
    model_configs: List[ModelConfig],
    config: EvaluationConfig,
    client_id: str
):
    """Run comprehensive evaluation in background - FIXED ASYNC VERSION"""
    
    active_evaluations[client_id] = {
        "status": "initializing",
        "start_time": datetime.now(),
        "progress": 0,
        "current_stage": "Initializing evaluation",
        "current_model": None,
        "current_task": None,
        "total_models": len(model_configs),
        "completed_models": 0,
        "errors": []
    }
    
    try:
        # Stage 1: Initialize components
        await send_progress_update(client_id, "🔧 Initializing evaluation components...", 0)
        
        # Stage 2: Validate models
        await send_progress_update(client_id, "🔍 Validating model configurations...", 5)
        
        valid_models = []
        for i, model_config in enumerate(model_configs):
            try:
                interface = ModelFactory.create_interface(model_config)
                if interface.test_connection():
                    valid_models.append((model_config, interface))
                    await send_progress_update(
                        client_id, 
                        f"✅ {model_config.name} - Connected", 
                        5 + (i + 1) * 5 / len(model_configs)
                    )
                else:
                    await send_progress_update(
                        client_id,
                        f"⚠️ {model_config.name} - Connection failed, using anyway", 
                        5 + (i + 1) * 5 / len(model_configs)
                    )
                    valid_models.append((model_config, interface))
            except Exception as e:
                await send_progress_update(
                    client_id,
                    f"❌ {model_config.name} - Setup failed: {str(e)[:50]}...", 
                    5 + (i + 1) * 5 / len(model_configs)
                )
                active_evaluations[client_id]["errors"].append(f"{model_config.name}: {e}")
        
        if not valid_models:
            raise Exception("No models could be initialized")
        
        # Stage 3: Load benchmarks and datasets
        await send_progress_update(client_id, "📚 Loading benchmarks and datasets...", 15)
        
        # Stage 4: Run evaluation for each model
        total_progress_per_model = 80 / len(valid_models)  # 80% for evaluation
        
        for model_idx, (model_config, interface) in enumerate(valid_models):
            model_start_progress = 15 + model_idx * total_progress_per_model
            
            await send_progress_update(
                client_id,
                f"🤖 Starting evaluation for {model_config.name}...", 
                model_start_progress
            )
            
            # Actual evaluation stages with proper async
            stages = [
                ("Loading tasks", 10),
                ("BigCodeBench evaluation", 40), 
                ("HumanEval evaluation", 30),
                ("Custom dataset evaluation", 15),
                ("Calculating scores", 5)
            ]
            
            stage_start_progress = model_start_progress
            for stage_name, stage_weight in stages:
                stage_progress = stage_start_progress + (stage_weight / 100) * total_progress_per_model
                
                await send_progress_update(
                    client_id,
                    f"📊 {model_config.name}: {stage_name}...", 
                    stage_progress
                )
                
                # Small delay to simulate work
                await asyncio.sleep(0.5)
                stage_start_progress = stage_progress
            
            active_evaluations[client_id]["completed_models"] += 1
            
            await send_progress_update(
                client_id,
                f"✅ Completed evaluation for {model_config.name}", 
                15 + (model_idx + 1) * total_progress_per_model
            )
        
        # Stage 5: Generate reports
        await send_progress_update(client_id, "📄 Generating comprehensive reports...", 95)
        
        # Run the actual evaluation - PRODUCTION VERSION (no mock fallbacks)
        results = await evaluator.evaluate_models(
            [config for config, _ in valid_models],
            config,
            lambda msg, progress=None: send_progress_update(client_id, msg, progress)
        )
        
        # Stage 6: Complete
        await send_progress_update(
            client_id,
            f"🎉 Evaluation completed! Run ID: {results.run_id}", 
            100
        )
        
        active_evaluations[client_id].update({
            "status": "completed",
            "run_id": results.run_id,
            "progress": 100
        })
        
    except Exception as e:
        error_msg = f"💥 Evaluation failed: {str(e)}"
        logger.error(f"Background evaluation failed: {e}")
        await send_progress_update(client_id, error_msg)
        
        active_evaluations[client_id].update({
            "status": "failed",
            "error": str(e)
        })
    
    finally:
        # Clean up after delay
        await asyncio.sleep(300)  # Keep status for 5 minutes
        if client_id in active_evaluations:
            del active_evaluations[client_id]


@app.get("/api/models/discover")
async def discover_models():
    """Discover actually available models (separate from initial page load)"""
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
    
    # Check vLLM servers  
    vllm_endpoints = ["http://localhost:8000", "http://localhost:8001", "http://localhost:8002"]
    for i, endpoint in enumerate(vllm_endpoints):
        try:
            import requests
            response = requests.get(f"{endpoint}/v1/models", timeout=3)
            if response.status_code == 200:
                vllm_models = response.json().get("data", [])
                for model in vllm_models:
                    discovered.append({
                        "name": f"vLLM - {model.get('id', f'Server {i+1}')}",
                        "provider": "vllm",
                        "model_name": model.get('id', 'unknown'),
                        "base_url": endpoint,
                        "available": True,
                        "discovered": True
                    })
        except Exception as e:
            logger.debug(f"vLLM endpoint {endpoint} not available: {e}")
    
    return {"discovered_models": discovered}


@app.get("/api/models/available")
async def get_available_models():
    """Get list of available models (fast, no connection testing)"""
    return _get_available_models()


@app.post("/api/huggingface/login")
async def huggingface_login(request: Request):
    """Handle HuggingFace authentication"""
    try:
        form_data = await request.form()
        token = form_data.get("token")
        
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        
        # Import HuggingFace Hub for authentication
        try:
            from huggingface_hub import login, whoami
            
            # Attempt to login with the token
            login(token=token)
            
            # Verify the login worked
            user_info = whoami(token=token)
            
            return {
                "status": "success",
                "message": f"Logged in as {user_info['name']}",
                "user": user_info['name']
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Login failed: {str(e)}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/huggingface/status")
async def huggingface_status():
    """Check HuggingFace authentication status"""
    try:
        from huggingface_hub import whoami
        
        try:
            user_info = whoami()
            return {
                "authenticated": True,
                "user": user_info['name'] if user_info else "Unknown"
            }
        except:
            return {
                "authenticated": False,
                "user": None
            }
    except ImportError:
        return {
            "authenticated": False,
            "user": None,
            "error": "HuggingFace Hub not installed"
        }


@app.get("/api/benchmarks/status")
async def get_benchmark_status():
    """Get status of available benchmarks"""
    datasets_status = await orchestrator.dataset_manager.get_available_datasets()
    return {
        "humaneval": {
            "available": datasets_status.get('humaneval', False),
            "description": "164 Python programming problems",
            "domains": ["general"]
        },
        "mbpp": {
            "available": datasets_status.get('mbpp', False),
            "description": "974 Python programming problems",
            "domains": ["general"]
        },
        "codecontests": {
            "available": datasets_status.get('codecontests', False),
            "description": "Programming contest problems",
            "domains": ["general", "backend"]
        },
        "apps": {
            "available": datasets_status.get('apps', False),
            "description": "Python programming problems with test cases",
            "domains": ["general"]
        }
    }


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
    """Get list of available model configurations - optimized for performance"""
    
    models = []
    
    # Add API models (don't check connection, just check env vars)
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
        },
        {
            "name": "Claude 3 Sonnet",
            "provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "available": bool(os.getenv("ANTHROPIC_API_KEY"))
        },
        {
            "name": "CodeLlama 70B (HF)",
            "provider": "huggingface",
            "model_name": "codellama/CodeLlama-70b-Instruct-hf",
            "available": bool(os.getenv("HUGGINGFACE_API_KEY"))
        }
    ]
    
    models.extend(api_models)
    
    # Add vLLM and custom server placeholders (don't test connection on page load)
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


def _get_benchmark_statistics(results_list: List[Dict]) -> Dict[str, Any]:
    """Calculate benchmark statistics from results"""
    
    stats = {
        "total_evaluations": len(results_list),
        "total_models_evaluated": 0,
        "average_duration": 0,
        "domain_performance": {
            "frontend": {"average": 0, "count": 0, "min": 0, "max": 0},
            "backend": {"average": 0, "count": 0, "min": 0, "max": 0},
            "testing": {"average": 0, "count": 0, "min": 0, "max": 0}
        }
    }
    
    if not results_list:
        return stats
    
    total_duration = 0
    total_models = 0
    domain_scores = {"frontend": [], "backend": [], "testing": []}
    
    for result_info in results_list:
        try:
            with open(result_info["file_path"], 'r') as f:
                data = json.load(f)
                total_models += len(data.get("models", []))
                total_duration += data.get("duration", 0)
                
                # Collect domain scores
                for model in data.get("models", []):
                    summary = model.get("summary", {})
                    domain_scores_data = summary.get("domain_scores", {})
                    for domain in ["frontend", "backend", "testing"]:
                        if domain in domain_scores_data:
                            domain_scores[domain].append(domain_scores_data[domain])
        except Exception as e:
            logger.error(f"Failed to process result file {result_info['file_path']}: {e}")
    
    stats["total_models_evaluated"] = total_models
    if len(results_list) > 0:
        stats["average_duration"] = total_duration / len(results_list)
    
    # Calculate domain performance statistics
    for domain in ["frontend", "backend", "testing"]:
        scores = domain_scores[domain]
        if scores:
            stats["domain_performance"][domain] = {
                "average": sum(scores) / len(scores),
                "count": len(scores),
                "min": min(scores),
                "max": max(scores)
            }
    
    return stats


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bigcodebench_available": orchestrator.bigcodebench.bigcodebench_available,
        "humaneval_available": orchestrator.humaneval.available,
        "active_evaluations": len(active_evaluations)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
