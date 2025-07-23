# src/web/app.py
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
import os
from datetime import datetime

# Import our evaluation components
from ..core.model_interfaces import ModelConfig, ModelFactory
from ..core.custom_datasets import DatasetManager, TaskType, DifficultyLevel
from ..evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Coding Evaluation Platform",
    description="Comprehensive evaluation platform for Large Language Models on coding tasks",
    version="1.0.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

# Initialize components
evaluation_engine = EvaluationEngine()
dataset_manager = DatasetManager()

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


class EvaluationRequest(BaseModel):
    model_configs: List[ModelConfigRequest]
    task_types: List[str] = ["frontend", "backend", "testing"]
    difficulty_levels: List[str] = ["easy", "medium"]
    max_tasks_per_type: int = 5
    include_bigcodebench: bool = False
    parallel_execution: bool = False


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "LLM Coding Evaluation Platform"}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    # Get recent evaluation results
    recent_results = evaluation_engine.list_evaluation_results()[:5]

    # Get dataset summary
    dataset_summary = dataset_manager.export_tasks_summary()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Dashboard",
            "recent_results": recent_results,
            "dataset_summary": dataset_summary,
            "active_evaluations": len(active_evaluations),
        },
    )


@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    """Evaluation configuration page"""
    # Get available model configurations
    default_models = [
        {
            "name": "CodeLlama 7B",
            "provider": "ollama",
            "model_name": "codellama:7b",
            "base_url": "http://localhost:11434",
        },
        {
            "name": "DeepSeek Coder 6.7B",
            "provider": "ollama",
            "model_name": "deepseek-coder:6.7b",
            "base_url": "http://localhost:11434",
        },
        {
            "name": "Qwen2.5 Coder 7B",
            "provider": "ollama",
            "model_name": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434",
        },
    ]

    return templates.TemplateResponse(
        "evaluate.html",
        {
            "request": request,
            "title": "Run Evaluation",
            "default_models": default_models,
            "task_types": [t.value for t in TaskType],
            "difficulty_levels": [d.value for d in DifficultyLevel],
        },
    )


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """Results listing page"""
    all_results = evaluation_engine.list_evaluation_results()

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "title": "Evaluation Results", "results": all_results},
    )


@app.get("/results/{run_id}", response_class=HTMLResponse)
async def result_detail(request: Request, run_id: str):
    """Detailed results for a specific run"""
    result_data = evaluation_engine.load_evaluation_results(run_id)

    if not result_data:
        raise HTTPException(status_code=404, detail="Results not found")

    return templates.TemplateResponse(
        "result_detail.html",
        {
            "request": request,
            "title": f"Results - {run_id[:8]}",
            "result_data": result_data,
            "run_id": run_id,
        },
    )


@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    """Dataset overview page"""
    dataset_summary = dataset_manager.export_tasks_summary()

    # Get sample tasks for each type
    frontend_tasks = dataset_manager.get_tasks_by_type(TaskType.FRONTEND)[:3]
    backend_tasks = dataset_manager.get_tasks_by_type(TaskType.BACKEND)[:3]
    testing_tasks = dataset_manager.get_tasks_by_type(TaskType.TESTING)[:3]

    return templates.TemplateResponse(
        "datasets.html",
        {
            "request": request,
            "title": "Datasets",
            "summary": dataset_summary,
            "frontend_tasks": frontend_tasks,
            "backend_tasks": backend_tasks,
            "testing_tasks": testing_tasks,
        },
    )


# API Endpoints
@app.post("/api/evaluate")
async def start_evaluation(
    request: EvaluationRequest, background_tasks: BackgroundTasks
):
    """Start a new evaluation run"""
    try:
        # Convert request to internal format
        model_configs = []
        for model_req in request.model_configs:
            config = ModelConfig(
                name=model_req.name,
                provider=model_req.provider,
                model_name=model_req.model_name,
                base_url=model_req.base_url,
                api_key=model_req.api_key,
                temperature=model_req.temperature,
                max_tokens=model_req.max_tokens,
            )
            model_configs.append(config)

        # Create evaluation config
        eval_config = EvaluationConfig(
            task_types=[TaskType(t) for t in request.task_types],
            difficulty_levels=[DifficultyLevel(d) for d in request.difficulty_levels],
            max_tasks_per_type=request.max_tasks_per_type,
            include_bigcodebench=request.include_bigcodebench,
            parallel_execution=request.parallel_execution,
        )

        # Generate run ID
        import uuid

        run_id = str(uuid.uuid4())

        # Store active evaluation info
        active_evaluations[run_id] = {
            "status": "starting",
            "started_at": datetime.now().isoformat(),
            "model_count": len(model_configs),
            "progress": "Initializing evaluation...",
        }

        # Start evaluation in background
        background_tasks.add_task(
            run_evaluation_background, run_id, model_configs, eval_config
        )

        return {
            "success": True,
            "run_id": run_id,
            "message": "Evaluation started successfully",
            "model_count": len(model_configs),
        }

    except Exception as e:
        logger.error(f"Failed to start evaluation: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/evaluation/{run_id}/status")
async def get_evaluation_status(run_id: str):
    """Get status of a running evaluation"""
    # Check active evaluations first
    if run_id in active_evaluations:
        return active_evaluations[run_id]

    # Check engine for status
    status = evaluation_engine.get_evaluation_status(run_id)
    if status:
        return status

    # Check if completed results exist
    result_data = evaluation_engine.load_evaluation_results(run_id)
    if result_data:
        return {
            "run_id": run_id,
            "status": result_data["status"],
            "completed_at": result_data["timestamp"],
            "duration": result_data["duration"],
        }

    raise HTTPException(status_code=404, detail="Evaluation not found")


@app.get("/api/models/test")
async def test_model_connection(
    provider: str,
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Test connection to a model"""
    try:
        config = ModelConfig(
            name=f"Test {model_name}",
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )

        interface = ModelFactory.create_interface(config)
        connected = interface.test_connection()

        return {
            "success": connected,
            "model_name": model_name,
            "provider": provider,
            "message": "Connected successfully" if connected else "Connection failed",
        }

    except Exception as e:
        return {
            "success": False,
            "model_name": model_name,
            "provider": provider,
            "message": f"Error: {str(e)}",
        }


@app.get("/api/datasets/summary")
async def get_dataset_summary():
    """Get dataset summary"""
    return dataset_manager.export_tasks_summary()


@app.get("/api/results")
async def list_results():
    """List all evaluation results"""
    return evaluation_engine.list_evaluation_results()


@app.get("/api/results/{run_id}")
async def get_results(run_id: str):
    """Get specific evaluation results"""
    result_data = evaluation_engine.load_evaluation_results(run_id)
    if not result_data:
        raise HTTPException(status_code=404, detail="Results not found")
    return result_data


@app.delete("/api/evaluation/{run_id}")
async def cancel_evaluation(run_id: str):
    """Cancel a running evaluation"""
    if run_id in active_evaluations:
        active_evaluations[run_id]["status"] = "cancelled"
        return {"success": True, "message": "Evaluation cancelled"}

    raise HTTPException(status_code=404, detail="Active evaluation not found")


# Background task function
async def run_evaluation_background(
    run_id: str, model_configs: List[ModelConfig], eval_config: EvaluationConfig
):
    """Run evaluation in background"""
    try:
        # Update status
        active_evaluations[run_id]["status"] = "running"

        def progress_callback(message: str):
            if run_id in active_evaluations:
                active_evaluations[run_id]["progress"] = message

        # Run evaluation
        result = await evaluation_engine.run_evaluation(
            model_configs, eval_config, progress_callback
        )

        # Update status
        active_evaluations[run_id].update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "duration": result.duration,
                "results_available": True,
            }
        )

        logger.info(f"Evaluation {run_id} completed successfully")

    except Exception as e:
        logger.error(f"Evaluation {run_id} failed: {e}")
        active_evaluations[run_id].update(
            {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            }
        )

    finally:
        # Clean up after some time
        async def cleanup():
            await asyncio.sleep(300)  # Keep for 5 minutes
            if run_id in active_evaluations:
                del active_evaluations[run_id]

        asyncio.create_task(cleanup())


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_evaluations": len(active_evaluations),
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "title": "Page Not Found",
            "error_code": 404,
            "error_message": "The requested page was not found.",
        },
        status_code=404,
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "title": "Server Error",
            "error_code": 500,
            "error_message": "An internal server error occurred.",
        },
        status_code=500,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.web.app:app", host="localhost", port=8000, reload=True, log_level="info"
    )
