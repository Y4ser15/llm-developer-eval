# src/evaluation/comprehensive_evaluator.py
"""
Comprehensive LLM Coding Evaluation Engine
Integrates multiple benchmarks and provides unified evaluation interface.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import uuid

from ..core.model_interfaces import ModelInterface, ModelConfig, ModelFactory
from ..core.bigcodebench_integration import BenchmarkOrchestrator, BigCodeBenchResult
from ..utils.report_generator import ReportGenerator


logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation"""
    # Domains to evaluate
    domains: List[str] = field(default_factory=lambda: ["frontend", "backend", "testing"])
    
    # Benchmark selection
    include_bigcodebench: bool = True
    include_humaneval: bool = True
    include_custom_datasets: bool = True
    
    # Task limits
    max_tasks_per_domain: int = 10
    max_total_tasks: int = 50
    
    # Execution settings
    parallel_models: bool = False
    timeout_per_task: int = 300
    
    # Output settings
    generate_report: bool = True
    save_results: bool = True
    export_format: str = "json"  # json, csv, html


@dataclass
class ModelEvaluationResult:
    """Complete evaluation results for a single model"""
    model_name: str
    provider: str
    model_config: Dict[str, Any]
    
    # Benchmark results
    bigcodebench_results: Dict[str, List[BigCodeBenchResult]] = field(default_factory=dict)
    humaneval_results: List[BigCodeBenchResult] = field(default_factory=list)
    custom_results: List[Any] = field(default_factory=list)
    
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
    """Main evaluation orchestrator for comprehensive LLM coding evaluation"""
    
    def __init__(self, results_dir: str = "results"):
        self.orchestrator = BenchmarkOrchestrator()
        self.report_generator = ReportGenerator()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Active evaluations
        self.active_runs: Dict[str, ComprehensiveEvaluationRun] = {}
    
    async def evaluate_models(
        self,
        model_configs: List[ModelConfig],
        config: EvaluationConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> ComprehensiveEvaluationRun:
        """Run comprehensive evaluation across multiple models"""
        
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
            logger.info(f"Starting comprehensive evaluation {run_id} with {len(model_configs)} models")
            
            if progress_callback:
                progress_callback(f"Starting evaluation of {len(model_configs)} models...")
            
            # Evaluate models
            if config.parallel_models:
                results = await self._evaluate_models_parallel(
                    model_configs, config, progress_callback
                )
            else:
                results = await self._evaluate_models_sequential(
                    model_configs, config, progress_callback
                )
            
            evaluation_run.model_results = results
            evaluation_run.total_duration = time.time() - start_time
            evaluation_run.status = "completed"
            
            # Generate report if requested
            if config.generate_report:
                await self._generate_comprehensive_report(evaluation_run)
            
            # Save results if requested
            if config.save_results:
                self._save_evaluation_results(evaluation_run, config.export_format)
            
            logger.info(f"Evaluation {run_id} completed in {evaluation_run.total_duration:.2f}s")
            
        except Exception as e:
            evaluation_run.status = "failed"
            evaluation_run.total_duration = time.time() - start_time
            logger.error(f"Evaluation {run_id} failed: {e}")
            raise
        
        finally:
            if run_id in self.active_runs:
                del self.active_runs[run_id]
        
        return evaluation_run
    
    async def _evaluate_models_sequential(
        self,
        model_configs: List[ModelConfig],
        config: EvaluationConfig,
        progress_callback: Optional[Callable]
    ) -> List[ModelEvaluationResult]:
        """Evaluate models sequentially"""
        
        results = []
        
        for i, model_config in enumerate(model_configs):
            if progress_callback:
                progress_callback(f"Evaluating model {i+1}/{len(model_configs)}: {model_config.name}")
            
            try:
                model_result = await self._evaluate_single_model(
                    model_config, config, progress_callback
                )
                results.append(model_result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_config.name}: {e}")
                # Create failed result
                failed_result = ModelEvaluationResult(
                    model_name=model_config.name,
                    provider=model_config.provider,
                    model_config=model_config.dict(),
                    errors=[str(e)]
                )
                results.append(failed_result)
        
        return results
    
    async def _evaluate_models_parallel(
        self,
        model_configs: List[ModelConfig],
        config: EvaluationConfig,
        progress_callback: Optional[Callable]
    ) -> List[ModelEvaluationResult]:
        """Evaluate models in parallel"""
        
        semaphore = asyncio.Semaphore(3)  # Limit concurrent evaluations
        
        async def evaluate_with_semaphore(model_config: ModelConfig) -> ModelEvaluationResult:
            async with semaphore:
                return await self._evaluate_single_model(model_config, config, progress_callback)
        
        # Create tasks
        tasks = [evaluate_with_semaphore(config) for config in model_configs]
        
        # Execute with exception handling
        results = []
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Model {model_configs[i].name} failed: {result}")
                failed_result = ModelEvaluationResult(
                    model_name=model_configs[i].name,
                    provider=model_configs[i].provider,
                    model_config=model_configs[i].dict(),
                    errors=[str(result)]
                )
                results.append(failed_result)
            else:
                results.append(result)
        
        return results
    
    async def _evaluate_single_model(
        self,
        model_config: ModelConfig,
        config: EvaluationConfig,
        progress_callback: Optional[Callable]
    ) -> ModelEvaluationResult:
        """Evaluate a single model comprehensively"""
        
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
            # BigCodeBench evaluation
            if config.include_bigcodebench:
                if progress_callback:
                    progress_callback(f"Running BigCodeBench evaluation for {model_config.name}")
                
                bigcodebench_results = await self.orchestrator.run_comprehensive_evaluation(
                    model_interface,
                    domains=config.domains,
                    include_humaneval=False,  # Handle separately
                    max_tasks_per_domain=config.max_tasks_per_domain,
                    progress_callback=progress_callback
                )
                result.bigcodebench_results = bigcodebench_results
            
            # HumanEval evaluation
            if config.include_humaneval:
                if progress_callback:
                    progress_callback(f"Running HumanEval evaluation for {model_config.name}")
                
                humaneval_results = await self.orchestrator.humaneval.evaluate_model(
                    model_interface,
                    max_tasks=config.max_tasks_per_domain
                )
                result.humaneval_results = humaneval_results
            
            # Calculate summary metrics
            self._calculate_summary_metrics(result)
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_config.name}: {e}")
            result.errors.append(str(e))
        
        result.execution_time = time.time() - start_time
        return result
    
    def _calculate_summary_metrics(self, result: ModelEvaluationResult):
        """Calculate summary metrics for model evaluation result"""
        
        all_results = []
        
        # Collect all benchmark results
        for domain_results in result.bigcodebench_results.values():
            all_results.extend(domain_results)
        all_results.extend(result.humaneval_results)
        
        if not all_results:
            return
        
        # Overall metrics
        result.total_tasks = len(all_results)
        result.passed_tasks = sum(1 for r in all_results if r.passed)
        result.overall_score = sum(r.score for r in all_results) / result.total_tasks
        
        # Domain-specific scores
        for domain in ["frontend", "backend", "testing"]:
            domain_key = f"bigcodebench_{domain}"
            if domain_key in result.bigcodebench_results:
                domain_results = result.bigcodebench_results[domain_key]
                if domain_results:
                    domain_score = sum(r.score for r in domain_results) / len(domain_results)
                    result.domain_scores[domain] = domain_score
        
        # HumanEval score
        if result.humaneval_results:
            humaneval_score = sum(r.score for r in result.humaneval_results) / len(result.humaneval_results)
            result.domain_scores["humaneval"] = humaneval_score
    
    async def _generate_comprehensive_report(self, evaluation_run: ComprehensiveEvaluationRun):
        """Generate comprehensive HTML report"""
        
        try:
            report_path = self.results_dir / f"comprehensive_report_{evaluation_run.run_id}.html"
            
            # Generate report using report generator
            await self.report_generator.generate_comprehensive_report(
                evaluation_run, str(report_path)
            )
            
            logger.info(f"Generated comprehensive report: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
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
            
            # Save in requested format
            if format == "json":
                with open(f"{base_path}.json", 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format == "csv":
                import pandas as pd
                
                # Create leaderboard CSV
                df = pd.DataFrame(export_data["leaderboard"])
                df.to_csv(f"{base_path}_leaderboard.csv", index=False)
                
                # Create detailed results CSV
                detailed_data = []
                for model in export_data["models"]:
                    row = {
                        "model_name": model["model_name"],
                        "provider": model["provider"],
                        **model["summary"]
                    }
                    detailed_data.append(row)
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_csv(f"{base_path}_detailed.csv", index=False)
            
            logger.info(f"Saved evaluation results: {base_path}.{format}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
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


# Example usage
if __name__ == "__main__":
    import asyncio
    from ..core.model_interfaces import ModelConfig
    
    async def main():
        # Create evaluator
        evaluator = ComprehensiveEvaluator()
        
        # Define models to evaluate
        model_configs = [
            ModelConfig(
                name="CodeLlama 7B",
                provider="ollama",
                model_name="codellama:7b"
            ),
            ModelConfig(
                name="DeepSeek Coder 6.7B",
                provider="ollama",
                model_name="deepseek-coder:6.7b"
            )
        ]
        
        # Configure evaluation
        config = EvaluationConfig(
            domains=["frontend", "backend", "testing"],
            max_tasks_per_domain=5,
            include_bigcodebench=True,
            include_humaneval=True,
            generate_report=True,
            save_results=True
        )
        
        # Progress callback
        def progress(message):
            print(f"📈 {message}")
        
        # Run evaluation
        print("🚀 Starting comprehensive LLM coding evaluation...")
        results = await evaluator.evaluate_models(
            model_configs, config, progress
        )
        
        # Print leaderboard
        print("\n🏆 LEADERBOARD:")
        leaderboard = results.get_leaderboard()
        for entry in leaderboard:
            print(f"{entry['rank']}. {entry['model_name']}: {entry['overall_score']:.3f}")
    
    asyncio.run(main())
