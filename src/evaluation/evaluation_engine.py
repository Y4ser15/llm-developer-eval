# src/evaluation/evaluation_engine.py
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
from ..core.custom_datasets import DatasetManager, Task, TaskType, DifficultyLevel
from ..core.bigcodebench_integration import (
    CustomBigCodeBenchRunner,
    BigCodeBenchResult,
    BigCodeBenchConfig,
)
from ..utils.report_generator import ReportGenerator
from ..core.custom_datasets import TaskType, DifficultyLevel


logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""

    # Task selection
    task_types: List[TaskType] = field(
        default_factory=lambda: [TaskType.FRONTEND, TaskType.BACKEND, TaskType.TESTING]
    )
    difficulty_levels: List[DifficultyLevel] = field(
        default_factory=lambda: [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
    )
    max_tasks_per_type: int = 5

    # Evaluation settings
    include_bigcodebench: bool = True
    bigcodebench_subset: str = "complete"
    bigcodebench_n_samples: int = 10

    # Performance settings
    parallel_execution: bool = False
    max_concurrent_evaluations: int = 3
    timeout_per_task: int = 300

    # Output settings
    save_generated_code: bool = True
    generate_detailed_report: bool = True
    export_results: bool = True


@dataclass
class TaskResult:
    """Result for a single task evaluation"""

    task_id: str
    task_title: str
    task_type: TaskType
    difficulty: DifficultyLevel
    model_name: str
    provider: str
    score: float
    passed: bool
    execution_time: float
    generated_code: str
    error_message: Optional[str] = None
    detailed_scores: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelResult:
    """Results for all tasks for a single model"""

    model_name: str
    provider: str
    model_config: Dict[str, Any]
    task_results: List[TaskResult] = field(default_factory=list)

    # Summary statistics
    total_tasks: int = 0
    passed_tasks: int = 0
    average_score: float = 0.0
    average_execution_time: float = 0.0

    # Breakdown by type
    frontend_results: List[TaskResult] = field(default_factory=list)
    backend_results: List[TaskResult] = field(default_factory=list)
    testing_results: List[TaskResult] = field(default_factory=list)
    bigcodebench_results: List[BigCodeBenchResult] = field(default_factory=list)

    def calculate_summary(self):
        """Calculate summary statistics"""
        if not self.task_results:
            return

        self.total_tasks = len(self.task_results)
        self.passed_tasks = sum(1 for r in self.task_results if r.passed)
        self.average_score = sum(r.score for r in self.task_results) / self.total_tasks
        self.average_execution_time = (
            sum(r.execution_time for r in self.task_results) / self.total_tasks
        )

        # Categorize results
        self.frontend_results = [
            r for r in self.task_results if r.task_type == TaskType.FRONTEND
        ]
        self.backend_results = [
            r for r in self.task_results if r.task_type == TaskType.BACKEND
        ]
        self.testing_results = [
            r for r in self.task_results if r.task_type == TaskType.TESTING
        ]


@dataclass
class EvaluationRun:
    """Complete evaluation run results"""

    run_id: str
    timestamp: datetime
    config: EvaluationConfig
    model_results: List[ModelResult] = field(default_factory=list)
    duration: float = 0.0
    status: str = "pending"  # pending, running, completed, failed

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Generate leaderboard from results"""
        leaderboard = []

        for model_result in self.model_results:
            leaderboard.append(
                {
                    "model_name": model_result.model_name,
                    "provider": model_result.provider,
                    "overall_score": model_result.average_score,
                    "pass_rate": model_result.passed_tasks
                    / max(model_result.total_tasks, 1),
                    "avg_time": model_result.average_execution_time,
                    "frontend_score": sum(
                        r.score for r in model_result.frontend_results
                    )
                    / max(len(model_result.frontend_results), 1),
                    "backend_score": sum(r.score for r in model_result.backend_results)
                    / max(len(model_result.backend_results), 1),
                    "testing_score": sum(r.score for r in model_result.testing_results)
                    / max(len(model_result.testing_results), 1),
                    "total_tasks": model_result.total_tasks,
                }
            )

        # Sort by overall score
        leaderboard.sort(key=lambda x: x["overall_score"], reverse=True)
        return leaderboard


class EvaluationEngine:
    """Main evaluation engine"""

    def __init__(self, results_dir: str = "results"):
        self.dataset_manager = DatasetManager()
        self.bigcodebench_runner = CustomBigCodeBenchRunner()
        self.report_generator = ReportGenerator()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Active runs
        self.active_runs: Dict[str, EvaluationRun] = {}

    async def run_evaluation(
        self,
        model_configs: List[ModelConfig],
        config: EvaluationConfig = None,
        progress_callback: Optional[Callable] = None,
    ) -> EvaluationRun:
        """Run complete evaluation for multiple models"""
        config = config or EvaluationConfig()
        run_id = str(uuid.uuid4())

        evaluation_run = EvaluationRun(
            run_id=run_id, timestamp=datetime.now(), config=config, status="running"
        )

        self.active_runs[run_id] = evaluation_run
        start_time = time.time()

        try:
            logger.info(
                f"Starting evaluation run {run_id} with {len(model_configs)} models"
            )

            # Get tasks to evaluate
            tasks = self._select_tasks(config)
            logger.info(f"Selected {len(tasks)} tasks for evaluation")

            # Evaluate each model
            for i, model_config in enumerate(model_configs):
                if progress_callback:
                    progress_callback(
                        f"Evaluating model {i+1}/{len(model_configs)}: {model_config.name}"
                    )

                try:
                    model_interface = ModelFactory.create_interface(model_config)
                    model_result = await self._evaluate_model(
                        model_interface, tasks, config, progress_callback
                    )
                    evaluation_run.model_results.append(model_result)

                except Exception as e:
                    logger.error(f"Failed to evaluate model {model_config.name}: {e}")
                    # Create empty result for failed model
                    model_result = ModelResult(
                        model_name=model_config.name,
                        provider=model_config.provider,
                        model_config=model_config.dict(),
                    )
                    evaluation_run.model_results.append(model_result)

            evaluation_run.duration = time.time() - start_time
            evaluation_run.status = "completed"

            # Generate report if requested
            if config.generate_detailed_report:
                await self._generate_evaluation_report(evaluation_run)

            # Save results
            if config.export_results:
                self._save_evaluation_results(evaluation_run)

            logger.info(
                f"Evaluation run {run_id} completed in {evaluation_run.duration:.2f}s"
            )

        except Exception as e:
            evaluation_run.status = "failed"
            evaluation_run.duration = time.time() - start_time
            logger.error(f"Evaluation run {run_id} failed: {e}")
            raise

        finally:
            if run_id in self.active_runs:
                del self.active_runs[run_id]

        return evaluation_run

    def _select_tasks(self, config: EvaluationConfig) -> List[Task]:
        """Select tasks based on configuration"""
        selected_tasks = []

        for task_type in config.task_types:
            # Get tasks for this type
            type_tasks = self.dataset_manager.get_tasks_by_type(task_type)

            # Filter by difficulty
            filtered_tasks = [
                task
                for task in type_tasks
                if task.difficulty in config.difficulty_levels
            ]

            # Limit number of tasks
            if config.max_tasks_per_type > 0:
                filtered_tasks = filtered_tasks[: config.max_tasks_per_type]

            selected_tasks.extend(filtered_tasks)

        return selected_tasks

    async def _evaluate_model(
        self,
        model_interface: ModelInterface,
        tasks: List[Task],
        config: EvaluationConfig,
        progress_callback: Optional[Callable] = None,
    ) -> ModelResult:
        """Evaluate a single model on all tasks"""
        model_result = ModelResult(
            model_name=model_interface.model_name,
            provider=model_interface.provider,
            model_config=model_interface.config.dict(),
        )

        try:
            # Test model connection first
            if not model_interface.test_connection():
                raise RuntimeError(
                    f"Cannot connect to model {model_interface.model_name}"
                )

            # Evaluate custom tasks
            if config.parallel_execution:
                task_results = await self._evaluate_tasks_parallel(
                    model_interface, tasks, config, progress_callback
                )
            else:
                task_results = await self._evaluate_tasks_sequential(
                    model_interface, tasks, config, progress_callback
                )

            model_result.task_results = task_results

            # Evaluate with BigCodeBench if requested
            if config.include_bigcodebench:
                try:
                    if progress_callback:
                        progress_callback(
                            f"Running BigCodeBench evaluation for {model_interface.model_name}"
                        )

                    bigcodebench_results = (
                        self.bigcodebench_runner.evaluate_with_bigcodebench(
                            model_interface,
                            subset=config.bigcodebench_subset,
                            n_samples=config.bigcodebench_n_samples,
                        )
                    )
                    model_result.bigcodebench_results = bigcodebench_results

                except Exception as e:
                    logger.warning(
                        f"BigCodeBench evaluation failed for {model_interface.model_name}: {e}"
                    )

            # Calculate summary statistics
            model_result.calculate_summary()

        except Exception as e:
            logger.error(
                f"Model evaluation failed for {model_interface.model_name}: {e}"
            )
            # Add error information to result
            model_result.task_results = []

        return model_result

    async def _evaluate_tasks_sequential(
        self,
        model_interface: ModelInterface,
        tasks: List[Task],
        config: EvaluationConfig,
        progress_callback: Optional[Callable] = None,
    ) -> List[TaskResult]:
        """Evaluate tasks sequentially"""
        results = []

        for i, task in enumerate(tasks):
            if progress_callback:
                progress_callback(f"Task {i+1}/{len(tasks)}: {task.title}")

            result = await self._evaluate_single_task(model_interface, task, config)
            results.append(result)

        return results

    async def _evaluate_tasks_parallel(
        self,
        model_interface: ModelInterface,
        tasks: List[Task],
        config: EvaluationConfig,
        progress_callback: Optional[Callable] = None,
    ) -> List[TaskResult]:
        """Evaluate tasks in parallel"""
        semaphore = asyncio.Semaphore(config.max_concurrent_evaluations)

        async def evaluate_with_semaphore(task: Task) -> TaskResult:
            async with semaphore:
                return await self._evaluate_single_task(model_interface, task, config)

        # Create tasks and run them
        coroutines = [evaluate_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[i].task_id} failed: {result}")
                # Create failed result
                final_results.append(
                    TaskResult(
                        task_id=tasks[i].task_id,
                        task_title=tasks[i].title,
                        task_type=tasks[i].task_type,
                        difficulty=tasks[i].difficulty,
                        model_name=model_interface.model_name,
                        provider=model_interface.provider,
                        score=0.0,
                        passed=False,
                        execution_time=0.0,
                        generated_code="",
                        error_message=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _evaluate_single_task(
        self, model_interface: ModelInterface, task: Task, config: EvaluationConfig
    ) -> TaskResult:
        """Evaluate a single task"""
        try:
            # Run evaluation using BigCodeBench runner
            bigcodebench_results = self.bigcodebench_runner.evaluate_custom_tasks(
                model_interface, [task]
            )

            if bigcodebench_results:
                bcb_result = bigcodebench_results[0]

                return TaskResult(
                    task_id=task.task_id,
                    task_title=task.title,
                    task_type=task.task_type,
                    difficulty=task.difficulty,
                    model_name=model_interface.model_name,
                    provider=model_interface.provider,
                    score=bcb_result.score,
                    passed=bcb_result.passed,
                    execution_time=bcb_result.execution_time,
                    generated_code=bcb_result.generated_code,
                    error_message=bcb_result.error_message,
                    detailed_scores=(
                        bcb_result.test_results.get("details", {})
                        if bcb_result.test_results
                        else None
                    ),
                )
            else:
                raise RuntimeError("No results returned from evaluation")

        except Exception as e:
            logger.error(f"Failed to evaluate task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                task_title=task.title,
                task_type=task.task_type,
                difficulty=task.difficulty,
                model_name=model_interface.model_name,
                provider=model_interface.provider,
                score=0.0,
                passed=False,
                execution_time=0.0,
                generated_code="",
                error_message=str(e),
            )

    async def _generate_evaluation_report(self, evaluation_run: EvaluationRun):
        """Generate detailed evaluation report"""
        try:
            report_path = self.results_dir / f"report_{evaluation_run.run_id}.html"

            # Use report generator to create detailed report
            await self.report_generator.generate_evaluation_report(
                evaluation_run, str(report_path)
            )

            logger.info(f"Generated evaluation report: {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

    def _save_evaluation_results(self, evaluation_run: EvaluationRun):
        """Save evaluation results to file"""
        try:
            # Save JSON results
            results_path = self.results_dir / f"results_{evaluation_run.run_id}.json"

            results_data = {
                "run_id": evaluation_run.run_id,
                "timestamp": evaluation_run.timestamp.isoformat(),
                "duration": evaluation_run.duration,
                "status": evaluation_run.status,
                "config": evaluation_run.config.__dict__,
                "leaderboard": evaluation_run.get_leaderboard(),
                "models": [],
            }

            for model_result in evaluation_run.model_results:
                model_data = {
                    "model_name": model_result.model_name,
                    "provider": model_result.provider,
                    "config": model_result.model_config,
                    "summary": {
                        "total_tasks": model_result.total_tasks,
                        "passed_tasks": model_result.passed_tasks,
                        "average_score": model_result.average_score,
                        "average_execution_time": model_result.average_execution_time,
                    },
                    "task_results": [
                        {
                            "task_id": r.task_id,
                            "task_title": r.task_title,
                            "task_type": r.task_type.value,
                            "difficulty": r.difficulty.value,
                            "score": r.score,
                            "passed": r.passed,
                            "execution_time": r.execution_time,
                            "error_message": r.error_message,
                            "detailed_scores": r.detailed_scores,
                        }
                        for r in model_result.task_results
                    ],
                }
                results_data["models"].append(model_data)

            with open(results_path, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info(f"Saved evaluation results: {results_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def get_evaluation_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running evaluation"""
        if run_id in self.active_runs:
            run = self.active_runs[run_id]
            return {
                "run_id": run_id,
                "status": run.status,
                "timestamp": run.timestamp.isoformat(),
                "completed_models": len(run.model_results),
                "duration": (
                    time.time() - run.timestamp.timestamp()
                    if run.status == "running"
                    else run.duration
                ),
            }
        return None

    def list_evaluation_results(self) -> List[Dict[str, Any]]:
        """List all evaluation results"""
        results = []

        for results_file in self.results_dir.glob("results_*.json"):
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                    results.append(
                        {
                            "run_id": data["run_id"],
                            "timestamp": data["timestamp"],
                            "duration": data["duration"],
                            "status": data["status"],
                            "model_count": len(data["models"]),
                            "file_path": str(results_file),
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to read results file {results_file}: {e}")

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    def load_evaluation_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load specific evaluation results"""
        results_file = self.results_dir / f"results_{run_id}.json"

        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load results {run_id}: {e}")

        return None


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create evaluation engine
        engine = EvaluationEngine()

        # Define model configurations
        model_configs = [
            ModelConfig(
                name="CodeLlama 7B",
                provider="ollama",
                model_name="codellama:7b",
                base_url="http://localhost:11434",
            ),
            ModelConfig(
                name="DeepSeek Coder 6.7B",
                provider="ollama",
                model_name="deepseek-coder:6.7b",
                base_url="http://localhost:11434",
            ),
        ]

        # Configure evaluation
        config = EvaluationConfig(
            task_types=[TaskType.FRONTEND, TaskType.BACKEND],
            difficulty_levels=[DifficultyLevel.EASY],
            max_tasks_per_type=2,
            include_bigcodebench=False,  # Skip for quick test
            parallel_execution=False,
        )

        # Run evaluation
        def progress_callback(message):
            print(f"Progress: {message}")

        results = await engine.run_evaluation(model_configs, config, progress_callback)

        # Print leaderboard
        print("\nLeaderboard:")
        leaderboard = results.get_leaderboard()
        for i, entry in enumerate(leaderboard, 1):
            print(f"{i}. {entry['model_name']}: {entry['overall_score']:.3f}")

    asyncio.run(main())
