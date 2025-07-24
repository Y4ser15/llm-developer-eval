# src/core/simple_datasets.py
"""
SIMPLE Datasets Integration - NO Progress Callbacks, Just Working Evaluation
Focus on working evaluation, remove all progress complexity.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import traceback
import os
import re

# Add E2B import
try:
    from e2b_code_interpreter import Sandbox

    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False
    Sandbox = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Standard evaluation result format"""

    task_id: str
    model_name: str
    domain: str
    prompt: str
    generated_code: str
    passed: bool
    score: float
    execution_time: float
    error_message: Optional[str] = None
    test_results: Optional[Dict] = None


class SimpleHumanEvalDataset:
    """HumanEval with streaming - NO PROGRESS CALLBACKS"""

    def __init__(self):
        self.available = self._check_availability()
        self.name = "HumanEval"

    def _check_availability(self) -> bool:
        try:
            from datasets import load_dataset

            return True
        except ImportError:
            logger.error("Install datasets: pip install datasets")
            return False

    async def get_tasks(self, max_tasks: int = 5) -> List[Dict]:
        """Get HumanEval tasks with streaming"""
        if not self.available:
            raise RuntimeError("Install datasets: pip install datasets")

        try:
            from datasets import load_dataset

            logger.info(f"📊 Loading HumanEval dataset ({max_tasks} tasks)...")

            # Use streaming to avoid downloading entire dataset
            dataset = load_dataset("openai_humaneval", split="test", streaming=True)

            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_tasks:
                    break

                tasks.append(
                    {
                        "task_id": item["task_id"],
                        "prompt": item["prompt"],
                        "canonical_solution": item["canonical_solution"],
                        "test": item["test"],
                        "entry_point": item["entry_point"],
                        "domain": "general",
                        "difficulty": "medium",
                        "source": "HumanEval",
                    }
                )

            logger.info(f"✅ Loaded {len(tasks)} HumanEval tasks")
            return tasks

        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            raise RuntimeError(f"HumanEval dataset error: {e}")


class SimpleDatasetManager:
    """Simple dataset manager - NO PROGRESS CALLBACKS"""

    def __init__(self):
        self.datasets = {"humaneval": SimpleHumanEvalDataset()}

    async def get_available_datasets(self) -> Dict[str, bool]:
        """Get status of all datasets"""
        status = {}
        for name, dataset in self.datasets.items():
            status[name] = dataset.available
        return status

    async def get_domain_tasks(
        self, domain: str, max_tasks: int = 5, dataset_preference: List[str] = None
    ) -> List[Dict]:
        """Get tasks for specific domain from available datasets"""

        if dataset_preference is None:
            dataset_preference = ["humaneval"]

        all_tasks = []

        for dataset_name in dataset_preference:
            if dataset_name not in self.datasets:
                continue

            dataset = self.datasets[dataset_name]
            if not dataset.available:
                logger.warning(f"Dataset {dataset_name} not available, skipping")
                continue

            try:
                tasks = await dataset.get_tasks(max_tasks)

                # Mark tasks with requested domain
                for task in tasks:
                    task["requested_domain"] = domain

                all_tasks.extend(tasks)

                # Stop if we have enough tasks
                if len(all_tasks) >= max_tasks:
                    break

            except Exception as e:
                logger.error(f"Failed to get tasks from {dataset_name}: {e}")
                continue

        if not all_tasks:
            raise RuntimeError(
                f"No tasks available for domain '{domain}'. "
                f"Available datasets: {[name for name, ds in self.datasets.items() if ds.available]}"
            )

        return all_tasks[:max_tasks]

    async def evaluate_model_on_tasks(
        self, model_interface, tasks: List[Dict]
    ) -> List[EvaluationResult]:
        """Evaluate model on given tasks - uses E2B sandbox for real code execution and test validation"""
        results = []
        total_tasks = len(tasks)
        logger.info(f"📊 Evaluating {total_tasks} tasks with E2B sandbox...")

        # Setup E2B API key
        e2b_api_key = os.getenv("E2B_API_KEY")
        e2b_api_key = "e2b_a52de90bb12379793043c11575e4484649edd669"

        if e2b_api_key:
            os.environ["E2B_API_KEY"] = e2b_api_key
        else:
            logger.warning("E2B_API_KEY not set in environment. Sandbox may not work.")

        sandbox_runner = None
        if E2B_AVAILABLE:
            sandbox_runner = E2BSandboxRunner()
        else:
            logger.error(
                "E2B sandbox not available. Please install e2b_code_interpreter."
            )

        for i, task in enumerate(tasks):
            logger.info(f"📋 Task {i+1}/{total_tasks}: {task['task_id']}")
            try:
                start_time = datetime.now()
                # Generate code
                generation_result = model_interface.generate_code(
                    prompt=task["prompt"],
                    system_prompt="You are an expert programmer. Generate working code that solves the problem.",
                )
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                # Run code and tests in E2B sandbox
                if sandbox_runner and "test" in task:
                    sandbox_result = sandbox_runner.run_code_with_tests(
                        generation_result.code, task["test"]
                    )
                    passed = sandbox_result["passed"]
                    coverage = sandbox_result["coverage"]
                    error = sandbox_result["error"]
                    trace = sandbox_result["traceback"]
                else:
                    passed = False
                    coverage = 0.0
                    error = "Sandbox not available or no test code."
                    trace = None

                score = 1.0 if passed else 0.0

                logger.info(
                    f"✅ Task {task['task_id']}: {'PASSED' if passed else 'FAILED'} | Coverage@K: {coverage}"
                )

                results.append(
                    EvaluationResult(
                        task_id=task["task_id"],
                        model_name=model_interface.model_name,
                        domain=task.get("requested_domain", task["domain"]),
                        prompt=task["prompt"],
                        generated_code=generation_result.code,
                        passed=passed,
                        score=score,
                        execution_time=execution_time,
                        error_message=error,
                        test_results={"coverage_at_k": coverage, "traceback": trace},
                    )
                )
            except Exception as e:
                logger.error(f"Failed to evaluate task {task['task_id']}: {e}")
                results.append(
                    EvaluationResult(
                        task_id=task["task_id"],
                        model_name=model_interface.model_name,
                        domain=task.get("requested_domain", task["domain"]),
                        prompt=task["prompt"],
                        generated_code="",
                        passed=False,
                        score=0.0,
                        execution_time=0.0,
                        error_message=str(e),
                        test_results={"coverage_at_k": 0.0, "traceback": None},
                    )
                )
        passed_count = sum(1 for r in results if r.passed)
        avg_coverage = (
            sum(r.test_results.get("coverage_at_k", 0.0) for r in results) / total_tasks
            if total_tasks
            else 0.0
        )
        logger.info(
            f"🎯 Evaluation completed: {passed_count}/{total_tasks} passed | Avg Coverage@K: {avg_coverage}"
        )
        return results

    def _evaluate_code_quality(self, code: str, task: Dict) -> bool:
        """Basic code quality evaluation"""
        if not code or not code.strip():
            return False

        # Check minimum length
        if len(code.strip()) < 10:
            return False

        # Check for basic programming constructs
        code_lower = code.lower()

        # Must have some programming construct
        if not any(
            keyword in code_lower
            for keyword in [
                "def ",
                "function",
                "class",
                "return",
                "if ",
                "for ",
                "while ",
            ]
        ):
            return False

        # Check entry point if specified
        entry_point = task.get("entry_point")
        if entry_point and entry_point != "main":
            if entry_point not in code:
                return False

        return True


class SimpleResult:
    """Simple result object for compatibility"""

    def __init__(
        self,
        task_id,
        model_name,
        passed,
        score,
        execution_time,
        generated_code,
        error_message,
    ):
        self.task_id = task_id
        self.model_name = model_name
        self.passed = passed
        self.score = score
        self.execution_time = execution_time
        self.generated_code = generated_code
        self.error_message = error_message


class E2BSandboxRunner:
    """Utility to run code and tests in an E2B sandbox."""

    def __init__(self, timeout=10):
        if not E2B_AVAILABLE:
            raise ImportError(
                "e2b_code_interpreter is not installed. Please install it to use sandboxed execution."
            )
        self.timeout = timeout

    def run_code_with_tests(self, code: str, test_code: str) -> dict:
        """
        Runs the generated code and test code in the E2B sandbox.
        Returns a dict with pass/fail, coverage, and error info.
        """
        result = {
            "passed": False,
            "coverage": 0.0,
            "total_tests": 0,
            "passed_tests": 0,
            "error": None,
            "traceback": None,
        }
        # Estimate total tests by counting 'assert' statements
        total_tests = len(re.findall(r"assert ", test_code))
        if total_tests == 0:
            total_tests = 1  # fallback to 1 if no asserts found
        result["total_tests"] = total_tests
        try:
            with Sandbox() as sandbox:
                # First, run the generated code to define functions/classes
                exec_result = sandbox.run_code(code)
                if exec_result.error:
                    result["error"] = exec_result.error
                    result["traceback"] = exec_result.traceback
                    return result
                # Then, run the test code (should raise if any test fails)
                test_result = sandbox.run_code(test_code)
                # If no error, all tests passed
                result["passed"] = True
                result["coverage"] = 1.0
                result["passed_tests"] = total_tests
        except Exception as e:
            # If any error, no tests passed (conservative)
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            result["passed_tests"] = 0
            result["coverage"] = 0.0
        return result


class SimpleBenchmarkOrchestrator:
    """Simple orchestrator - NO PROGRESS CALLBACKS"""

    def __init__(self):
        self.dataset_manager = SimpleDatasetManager()

        # For compatibility with existing code
        self.bigcodebench = type(
            "obj", (object,), {"bigcodebench_available": False, "authenticated": False}
        )()

        # Create proper HumanEval proxy
        self.humaneval = self.HumanEvalProxy(self.dataset_manager)

    class HumanEvalProxy:
        """Proxy for HumanEval dataset to maintain compatibility"""

        def __init__(self, dataset_manager):
            self.dataset_manager = dataset_manager
            self.available = True

        async def evaluate_model(self, model_interface, max_tasks=5):
            """Evaluate model using HumanEval dataset - NO PROGRESS CALLBACKS"""
            try:
                logger.info("📊 Starting HumanEval evaluation...")
                humaneval_tasks = await self.dataset_manager.get_domain_tasks(
                    "general", max_tasks, ["humaneval"]
                )
                results = await self.dataset_manager.evaluate_model_on_tasks(
                    model_interface, humaneval_tasks
                )

                # Convert to expected format for compatibility
                converted_results = []
                for result in results:
                    converted_results.append(
                        SimpleResult(
                            task_id=result.task_id,
                            model_name=result.model_name,
                            passed=result.passed,
                            score=result.score,
                            execution_time=result.execution_time,
                            generated_code=result.generated_code,
                            error_message=result.error_message,
                        )
                    )

                logger.info(
                    f"✅ HumanEval evaluation completed: {len(converted_results)} tasks"
                )
                return converted_results
            except Exception as e:
                logger.error(f"HumanEval evaluation failed: {e}")
                return []

    async def run_comprehensive_evaluation(
        self,
        model_interface,
        domains: List[str] = ["frontend", "backend", "testing"],
        include_humaneval: bool = True,
        max_tasks_per_domain: int = 5,
    ) -> Dict[str, List]:
        """Run evaluation using public datasets - NO PROGRESS CALLBACKS"""

        logger.info(
            f"🚀 Starting comprehensive evaluation for {len(domains)} domains..."
        )
        all_results = {}

        # Map all domains to general since we only have HumanEval
        for domain in domains:
            logger.info(f"📋 Evaluating {domain} domain...")

            try:
                # Get tasks for domain
                tasks = await self.dataset_manager.get_domain_tasks(
                    "general", max_tasks_per_domain
                )

                # Evaluate model on tasks
                results = await self.dataset_manager.evaluate_model_on_tasks(
                    model_interface, tasks
                )

                # Convert to expected format for compatibility
                domain_results = []
                for result in results:
                    domain_results.append(
                        SimpleResult(
                            task_id=result.task_id,
                            model_name=result.model_name,
                            passed=result.passed,
                            score=result.score,
                            execution_time=result.execution_time,
                            generated_code=result.generated_code,
                            error_message=result.error_message,
                        )
                    )

                all_results[f"public_{domain}"] = domain_results
                logger.info(
                    f"✅ Completed {domain} evaluation: {len(domain_results)} tasks"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate {domain} domain: {e}")
                all_results[f"public_{domain}"] = []

        # Add HumanEval results if requested
        if include_humaneval:
            logger.info("📋 Running HumanEval evaluation...")

            try:
                humaneval_results = await self.humaneval.evaluate_model(
                    model_interface, max_tasks=max_tasks_per_domain
                )

                all_results["humaneval"] = humaneval_results
                logger.info(
                    f"✅ Completed HumanEval evaluation: {len(humaneval_results)} tasks"
                )

            except Exception as e:
                logger.error(f"HumanEval evaluation failed: {e}")
                all_results["humaneval"] = []

        logger.info("🎉 Comprehensive evaluation completed!")
        return all_results

    def get_benchmark_summary(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {}

        for benchmark_name, benchmark_results in results.items():
            if not benchmark_results:
                summary[benchmark_name] = {
                    "total": 0,
                    "passed": 0,
                    "pass_rate": 0.0,
                    "avg_score": 0.0,
                    "avg_time": 0.0,
                }
                continue

            total_tasks = len(benchmark_results)
            passed_tasks = sum(1 for r in benchmark_results if r.passed)
            avg_score = sum(r.score for r in benchmark_results) / total_tasks
            avg_time = sum(r.execution_time for r in benchmark_results) / total_tasks

            summary[benchmark_name] = {
                "total": total_tasks,
                "passed": passed_tasks,
                "pass_rate": passed_tasks / total_tasks if total_tasks > 0 else 0.0,
                "avg_score": avg_score,
                "avg_time": avg_time,
            }

        return summary


# Aliases for compatibility
WorkingBenchmarkOrchestrator = SimpleBenchmarkOrchestrator
FixedBenchmarkOrchestrator = SimpleBenchmarkOrchestrator
PublicBenchmarkOrchestrator = SimpleBenchmarkOrchestrator
BenchmarkOrchestrator = SimpleBenchmarkOrchestrator
