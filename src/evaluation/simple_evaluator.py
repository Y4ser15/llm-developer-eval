# src/evaluation/simple_evaluator.py
"""
SIMPLE LLM Coding Evaluation Engine - NO PROGRESS CALLBACKS
Focus on working evaluation without progress complexity.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import uuid

from ..core.model_interfaces import ModelInterface, ModelConfig, ModelFactory
from ..core.simple_datasets import SimpleBenchmarkOrchestrator, EvaluationResult
from ..utils.report_generator import ReportGenerator


logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation"""
    # Domains to evaluate
    domains: List[str] = field(default_factory=lambda: ["frontend", "backend", "testing"])
    
    # Benchmark selection
    include_humaneval: bool = True
    
    # Task limits
    max_tasks_per_domain: int = 5
    max_total_tasks: int = 50
    
    # Execution settings
    parallel_models: bool = False
    timeout_per_task: int = 300
    
    # Output settings
    generate_report: bool = True
    save_results: bool = True
    export_format: str = "json"


@dataclass
class ModelEvaluationResult:
    """Complete evaluation results for a single model"""
    model_name: str
    provider: str
    model_config: Dict[str, Any]
    
    # Benchmark results
    domain_results: Dict[str, List[EvaluationResult]] = field(default_factory=dict)
    humaneval_results: List[EvaluationResult] = field(default_factory=list)
    
    # Summary metrics
    total_tasks: int = 0
    passed_tasks: int = 0
    overall_score: float = 0.0
    domain_scores: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    
    # Metadata
    evaluation_time: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveEvaluationRun:
    """Complete evaluation run across multiple models"""
    run_id: str
    timestamp: datetime
    config: EvaluationConfig
    model_results: List[ModelEvaluationResult] = field(default_factory=list)
    
    # Summary statistics
    total_duration: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Generate leaderboard from results"""
        leaderboard = []
        
        for result in self.model_results:
            entry = {
                "model_name": result.model_name,
                "provider": result.provider,
                "overall_score": result.overall_score,
                "pass_rate": result.passed_tasks / max(result.total_tasks, 1),
                "total_tasks": result.total_tasks,
                "execution_time": result.execution_time,
                "domain_scores": result.domain_scores,
                "rank": 0  # Will be set after sorting
            }
            leaderboard.append(entry)
        
        # Sort by overall score
        leaderboard.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Add ranks
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        return leaderboard


class ComprehensiveEvaluator:
    """SIMPLE evaluation orchestrator - NO PROGRESS CALLBACKS"""
    
    def __init__(self, results_dir: str = "results"):
        self.orchestrator = SimpleBenchmarkOrchestrator()
        self.report_generator = ReportGenerator()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Active evaluations
        self.active_runs: Dict[str, ComprehensiveEvaluationRun] = {}
    
    async def evaluate_models(
        self,
        model_configs: List[ModelConfig],
        config: EvaluationConfig = None
    ) -> ComprehensiveEvaluationRun:
        """Run comprehensive evaluation across multiple models - NO PROGRESS CALLBACKS"""
        
        config = config or EvaluationConfig()
        run_id = str(uuid.uuid4())
        
        evaluation_run = ComprehensiveEvaluationRun(
            run_id=run_id,
            timestamp=datetime.now(),
            config=config,
            status="running"
        )
        
        self.active_runs[run_id] = evaluation_run
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting comprehensive evaluation {run_id} with {len(model_configs)} models")
            
            # Evaluate models sequentially (no parallel for simplicity)
            results = await self._evaluate_models_sequential(model_configs, config)
            
            evaluation_run.model_results = results
            evaluation_run.total_duration = time.time() - start_time
            evaluation_run.status = "completed"
            
            # Generate report if requested
            if config.generate_report:
                await self._generate_comprehensive_report(evaluation_run)
            
            # Save results if requested
            if config.save_results:
                self._save_evaluation_results(evaluation_run, config.export_format)
            
            logger.info(f"✅ Evaluation {run_id} completed in {evaluation_run.total_duration:.2f}s")
            
        except Exception as e:
            evaluation_run.status = "failed"
            evaluation_run.total_duration = time.time() - start_time
            logger.error(f"❌ Evaluation {run_id} failed: {e}")
            raise
        
        finally:
            if run_id in self.active_runs:
                del self.active_runs[run_id]
        
        return evaluation_run
    
    async def _evaluate_models_sequential(
        self,
        model_configs: List[ModelConfig],
        config: EvaluationConfig
    ) -> List[ModelEvaluationResult]:
        """Evaluate models sequentially - NO PROGRESS CALLBACKS"""
        
        results = []
        
        for i, model_config in enumerate(model_configs):
            logger.info(f"📋 Evaluating model {i+1}/{len(model_configs)}: {model_config.name}")
            
            try:
                model_result = await self._evaluate_single_model(model_config, config)
                results.append(model_result)
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate model {model_config.name}: {e}")
                # Create failed result
                failed_result = ModelEvaluationResult(
                    model_name=model_config.name,
                    provider=model_config.provider,
                    model_config=model_config.dict(),
                    errors=[str(e)]
                )
                results.append(failed_result)
        
        return results
    
    async def _evaluate_single_model(
        self,
        model_config: ModelConfig,
        config: EvaluationConfig
    ) -> ModelEvaluationResult:
        """Evaluate a single model comprehensively - NO PROGRESS CALLBACKS"""
        
        start_time = time.time()
        
        # Create model interface
        model_interface = ModelFactory.create_interface(model_config)
        
        # Test connection
        if not model_interface.test_connection():
            raise RuntimeError(f"Cannot connect to model {model_config.name}")
        
        result = ModelEvaluationResult(
            model_name=model_config.name,
            provider=model_config.provider,
            model_config=model_config.dict()
        )
        
        try:
            logger.info(f"🔍 Running comprehensive evaluation for {model_config.name}")
            
            # Run comprehensive evaluation using our simple orchestrator
            evaluation_results = await self.orchestrator.run_comprehensive_evaluation(
                model_interface,
                domains=config.domains,
                include_humaneval=config.include_humaneval,
                max_tasks_per_domain=config.max_tasks_per_domain
            )
            
            # Store results
            result.domain_results = evaluation_results
            
            # Calculate summary metrics
            self._calculate_summary_metrics(result)
            
            logger.info(f"✅ Completed evaluation for {model_config.name}: {result.passed_tasks}/{result.total_tasks} passed")
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_config.name}: {e}")
            result.errors.append(str(e))
        
        result.execution_time = time.time() - start_time
        return result
    
    def _calculate_summary_metrics(self, result: ModelEvaluationResult):
        """Calculate summary metrics for model evaluation result"""
        
        all_results = []
        
        # Collect all results
        for domain_name, domain_results in result.domain_results.items():
            all_results.extend(domain_results)
        
        if not all_results:
            return
        
        # Overall metrics
        result.total_tasks = len(all_results)
        result.passed_tasks = sum(1 for r in all_results if r.passed)
        result.overall_score = sum(r.score for r in all_results) / result.total_tasks
        
        # Domain-specific scores
        for domain_name, domain_results in result.domain_results.items():
            if domain_results:
                domain_score = sum(r.score for r in domain_results) / len(domain_results)
                # Clean up domain name for display
                clean_domain = domain_name.replace("public_", "")
                result.domain_scores[clean_domain] = domain_score
    
    async def _generate_comprehensive_report(self, evaluation_run: ComprehensiveEvaluationRun):
        """Generate comprehensive HTML report"""
        
        try:
            report_path = self.results_dir / f"comprehensive_report_{evaluation_run.run_id}.html"
            
            # Generate report using report generator
            await self.report_generator.generate_comprehensive_report(
                evaluation_run, str(report_path)
            )
            
            logger.info(f"📊 Generated comprehensive report: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate report: {e}")
    
    def _save_evaluation_results(self, evaluation_run: ComprehensiveEvaluationRun, format: str = "json"):
        """Save evaluation results in specified format"""
        
        try:
            base_path = self.results_dir / f"comprehensive_results_{evaluation_run.run_id}"
            
            # Prepare data for export
            export_data = {
                "run_id": evaluation_run.run_id,
                "timestamp": evaluation_run.timestamp.isoformat(),
                "config": evaluation_run.config.__dict__,
                "duration": evaluation_run.total_duration,
                "status": evaluation_run.status,
                "leaderboard": evaluation_run.get_leaderboard(),
                "models": []
            }
            
            for model_result in evaluation_run.model_results:
                model_data = {
                    "model_name": model_result.model_name,
                    "provider": model_result.provider,
                    "config": model_result.model_config,
                    "summary": {
                        "total_tasks": model_result.total_tasks,
                        "passed_tasks": model_result.passed_tasks,
                        "overall_score": model_result.overall_score,
                        "domain_scores": model_result.domain_scores,
                        "execution_time": model_result.execution_time
                    },
                    "errors": model_result.errors
                }
                export_data["models"].append(model_data)
            
            # Save in JSON format
            if format == "json":
                with open(f"{base_path}.json", 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"💾 Saved evaluation results: {base_path}.{format}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def get_evaluation_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of running evaluation"""
        
        if run_id in self.active_runs:
            run = self.active_runs[run_id]
            return {
                "run_id": run_id,
                "status": run.status,
                "timestamp": run.timestamp.isoformat(),
                "completed_models": len(run.model_results),
                "duration": time.time() - run.timestamp.timestamp()
            }
        return None
    
    def list_evaluation_results(self) -> List[Dict[str, Any]]:
        """List all saved evaluation results"""
        
        results = []
        
        for result_file in self.results_dir.glob("comprehensive_results_*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results.append({
                        "run_id": data["run_id"],
                        "timestamp": data["timestamp"],
                        "duration": data["duration"],
                        "status": data["status"],
                        "model_count": len(data["models"]),
                        "file_path": str(result_file)
                    })
            except Exception as e:
                logger.error(f"Failed to read result file {result_file}: {e}")
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results
