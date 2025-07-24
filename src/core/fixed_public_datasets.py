# src/core/fixed_public_datasets.py
"""
FIXED Public Datasets Integration - Streaming, No Full Downloads, Proper Async
Uses well-known public datasets that work immediately without authentication.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

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


class StreamingHumanEvalDataset:
    """HumanEval with streaming - no full download"""
    
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
    
    async def get_tasks(self, max_tasks: int = 10) -> List[Dict]:
        """Get HumanEval tasks with streaming - FIXED VERSION"""
        if not self.available:
            raise RuntimeError("Install datasets: pip install datasets")
        
        try:
            from datasets import load_dataset
            logger.info(f"📊 Loading HumanEval dataset (streaming {max_tasks} tasks)...")
            
            # Use streaming to avoid downloading entire dataset
            dataset = load_dataset("openai_humaneval", split="test", streaming=True)
            
            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_tasks:
                    break
                    
                tasks.append({
                    'task_id': item['task_id'],
                    'prompt': item['prompt'],
                    'canonical_solution': item['canonical_solution'],
                    'test': item['test'],
                    'entry_point': item['entry_point'],
                    'domain': 'general',
                    'difficulty': 'medium',
                    'source': 'HumanEval'
                })
            
            logger.info(f"✅ Loaded {len(tasks)} HumanEval tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            raise RuntimeError(f"HumanEval dataset error: {e}")


class LightweightMBPPDataset:
    """MBPP with limited download to avoid issues"""
    
    def __init__(self):
        self.available = self._check_availability()
        self.name = "MBPP"
    
    def _check_availability(self) -> bool:
        try:
            from datasets import load_dataset
            return True
        except ImportError:
            return False
    
    async def get_tasks(self, max_tasks: int = 10) -> List[Dict]:
        """Get MBPP tasks - LIGHTWEIGHT VERSION"""
        if not self.available:
            raise RuntimeError("Install datasets: pip install datasets")
        
        try:
            from datasets import load_dataset
            logger.info(f"📊 Loading MBPP dataset (streaming {max_tasks} tasks)...")
            
            # Use streaming to avoid full download
            dataset = load_dataset("mbpp", "sanitized", split="test", streaming=True)
            
            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_tasks:
                    break
                    
                tasks.append({
                    'task_id': f"MBPP/{item['task_id']}",
                    'prompt': item['text'],
                    'canonical_solution': item['code'],
                    'test': item['test_list'][0] if item['test_list'] else '',
                    'entry_point': 'solution',
                    'domain': 'general',
                    'difficulty': 'basic',
                    'source': 'MBPP'
                })
            
            logger.info(f"✅ Loaded {len(tasks)} MBPP tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load MBPP: {e}")
            # Return some basic programming tasks as fallback
            return await self._get_basic_python_tasks(max_tasks)
    
    async def _get_basic_python_tasks(self, max_tasks: int) -> List[Dict]:
        """Fallback basic Python tasks"""
        basic_tasks = [
            {
                'task_id': 'MBPP/basic_001',
                'prompt': 'Write a function to find the sum of two numbers.',
                'canonical_solution': 'def add_numbers(a, b):\\n    return a + b',
                'test': 'assert add_numbers(2, 3) == 5',
                'entry_point': 'add_numbers',
                'domain': 'general',
                'difficulty': 'basic',
                'source': 'MBPP_fallback'
            },
            {
                'task_id': 'MBPP/basic_002',
                'prompt': 'Write a function to check if a number is even.',
                'canonical_solution': 'def is_even(n):\\n    return n % 2 == 0',
                'test': 'assert is_even(4) == True',
                'entry_point': 'is_even',
                'domain': 'general',
                'difficulty': 'basic',
                'source': 'MBPP_fallback'
            }
        ]
        return basic_tasks[:max_tasks]


class FixedPublicDatasetManager:
    """FIXED dataset manager with streaming and proper async"""
    
    def __init__(self):
        self.datasets = {
            'humaneval': StreamingHumanEvalDataset(),
            'mbpp': LightweightMBPPDataset()
        }
    
    async def get_available_datasets(self) -> Dict[str, bool]:
        """Get status of all datasets"""
        status = {}
        for name, dataset in self.datasets.items():
            status[name] = dataset.available
        return status
    
    async def get_domain_tasks(
        self, 
        domain: str, 
        max_tasks: int = 10,
        dataset_preference: List[str] = None
    ) -> List[Dict]:
        """Get tasks for specific domain from available datasets"""
        
        if dataset_preference is None:
            dataset_preference = ['humaneval', 'mbpp']
        
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
                    task['requested_domain'] = domain
                    
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
        self,
        model_interface,
        tasks: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> List[EvaluationResult]:
        """Evaluate model on given tasks - FIXED ASYNC"""
        
        results = []
        total_tasks = len(tasks)
        
        for i, task in enumerate(tasks):
            # FIXED: Proper async handling for progress callback
            if progress_callback:
                try:
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(f"Task {i+1}/{total_tasks}: {task['task_id']}")
                    else:
                        progress_callback(f"Task {i+1}/{total_tasks}: {task['task_id']}")
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")
            
            try:
                start_time = datetime.now()
                
                # Generate code
                generation_result = model_interface.generate_code(
                    prompt=task['prompt'],
                    system_prompt="You are an expert programmer. Generate working code that solves the problem."
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Basic evaluation
                passed = self._evaluate_code_quality(generation_result.code, task)
                score = 1.0 if passed else 0.0
                
                results.append(EvaluationResult(
                    task_id=task['task_id'],
                    model_name=model_interface.model_name,
                    domain=task.get('requested_domain', task['domain']),
                    prompt=task['prompt'],
                    generated_code=generation_result.code,
                    passed=passed,
                    score=score,
                    execution_time=execution_time,
                    error_message=generation_result.error
                ))
                
            except Exception as e:
                logger.error(f"Failed to evaluate task {task['task_id']}: {e}")
                results.append(EvaluationResult(
                    task_id=task['task_id'],
                    model_name=model_interface.model_name,
                    domain=task.get('requested_domain', task['domain']),
                    prompt=task['prompt'],
                    generated_code="",
                    passed=False,
                    score=0.0,
                    execution_time=0.0,
                    error_message=str(e)
                ))
        
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
        if not any(keyword in code_lower for keyword in ['def ', 'function', 'class', 'return', 'if ', 'for ', 'while ']):
            return False
        
        # Check entry point if specified
        entry_point = task.get('entry_point')
        if entry_point and entry_point != 'main':
            if entry_point not in code:
                return False
        
        return True


class FixedBenchmarkOrchestrator:
    """FIXED orchestrator with proper async and streaming datasets"""
    
    def __init__(self):
        self.dataset_manager = FixedPublicDatasetManager()
        
        # For compatibility with existing code
        self.bigcodebench = type('obj', (object,), {
            'bigcodebench_available': False,
            'authenticated': False
        })()
        
        # Create proper HumanEval proxy
        self.humaneval = self.HumanEvalProxy(self.dataset_manager)
    
    class HumanEvalProxy:
        """Proxy for HumanEval dataset to maintain compatibility"""
        
        def __init__(self, dataset_manager):
            self.dataset_manager = dataset_manager
            self.available = True
        
        async def evaluate_model(self, model_interface, max_tasks=10, progress_callback=None):
            """Evaluate model using HumanEval dataset - FIXED ASYNC"""
            try:
                humaneval_tasks = await self.dataset_manager.get_domain_tasks('general', max_tasks, ['humaneval'])
                results = await self.dataset_manager.evaluate_model_on_tasks(
                    model_interface, humaneval_tasks, progress_callback
                )
                
                # Convert to expected format for compatibility
                converted_results = []
                for result in results:
                    # Create a simple result object
                    class SimpleResult:
                        def __init__(self, task_id, model_name, passed, score, execution_time, generated_code, error_message):
                            self.task_id = task_id
                            self.model_name = model_name
                            self.passed = passed
                            self.score = score
                            self.execution_time = execution_time
                            self.generated_code = generated_code
                            self.error_message = error_message
                    
                    converted_results.append(SimpleResult(
                        task_id=result.task_id,
                        model_name=result.model_name,
                        passed=result.passed,
                        score=result.score,
                        execution_time=result.execution_time,
                        generated_code=result.generated_code,
                        error_message=result.error_message
                    ))\n                \n                return converted_results\n            except Exception as e:\n                logger.error(f\"HumanEval evaluation failed: {e}\")\n                return []\n    \n    async def run_comprehensive_evaluation(\n        self,\n        model_interface,\n        domains: List[str] = [\"frontend\", \"backend\", \"testing\"],\n        include_humaneval: bool = True,\n        max_tasks_per_domain: int = 5,  # REDUCED default to avoid long downloads\n        progress_callback: Optional[Callable] = None\n    ) -> Dict[str, List]:\n        \"\"\"Run evaluation using public datasets - FIXED ASYNC VERSION\"\"\"\n        \n        all_results = {}\n        \n        # Map domains to appropriate datasets\n        domain_mapping = {\n            'frontend': 'general',\n            'backend': 'general', \n            'testing': 'general'\n        }\n        \n        for domain in domains:\n            # FIXED: Proper async handling for progress callback\n            if progress_callback:\n                try:\n                    if asyncio.iscoroutinefunction(progress_callback):\n                        await progress_callback(f\"Evaluating {domain} domain...\")\n                    else:\n                        progress_callback(f\"Evaluating {domain} domain...\")\n                except Exception as e:\n                    logger.debug(f\"Progress callback error: {e}\")\n            \n            try:\n                # Get tasks for domain\n                mapped_domain = domain_mapping.get(domain, 'general')\n                tasks = await self.dataset_manager.get_domain_tasks(\n                    mapped_domain,\n                    max_tasks_per_domain\n                )\n                \n                # Evaluate model on tasks\n                results = await self.dataset_manager.evaluate_model_on_tasks(\n                    model_interface,\n                    tasks,\n                    progress_callback\n                )\n                \n                # Convert to expected format for compatibility\n                domain_results = []\n                for result in results:\n                    # Create a simple result object\n                    class SimpleResult:\n                        def __init__(self, task_id, model_name, passed, score, execution_time, generated_code, error_message):\n                            self.task_id = task_id\n                            self.model_name = model_name\n                            self.passed = passed\n                            self.score = score\n                            self.execution_time = execution_time\n                            self.generated_code = generated_code\n                            self.error_message = error_message\n                    \n                    domain_results.append(SimpleResult(\n                        task_id=result.task_id,\n                        model_name=result.model_name,\n                        passed=result.passed,\n                        score=result.score,\n                        execution_time=result.execution_time,\n                        generated_code=result.generated_code,\n                        error_message=result.error_message\n                    ))\n                \n                all_results[f\"public_{domain}\"] = domain_results\n                logger.info(f\"Completed {domain} evaluation: {len(domain_results)} tasks\")\n                \n            except Exception as e:\n                logger.error(f\"Failed to evaluate {domain} domain: {e}\")\n                all_results[f\"public_{domain}\"] = []\n        \n        # Add HumanEval results if requested\n        if include_humaneval:\n            # FIXED: Proper async handling for progress callback\n            if progress_callback:\n                try:\n                    if asyncio.iscoroutinefunction(progress_callback):\n                        await progress_callback(\"Evaluating with HumanEval...\")\n                    else:\n                        progress_callback(\"Evaluating with HumanEval...\")\n                except Exception as e:\n                    logger.debug(f\"Progress callback error: {e}\")\n            \n            try:\n                humaneval_results = await self.humaneval.evaluate_model(\n                    model_interface,\n                    max_tasks=max_tasks_per_domain,\n                    progress_callback=progress_callback\n                )\n                \n                all_results[\"humaneval\"] = humaneval_results\n                logger.info(f\"Completed HumanEval evaluation: {len(humaneval_results)} tasks\")\n                \n            except Exception as e:\n                logger.error(f\"HumanEval evaluation failed: {e}\")\n                all_results[\"humaneval\"] = []\n        \n        return all_results\n    \n    def get_benchmark_summary(self, results: Dict[str, List]) -> Dict[str, Any]:\n        \"\"\"Generate summary statistics\"\"\"\n        summary = {}\n        \n        for benchmark_name, benchmark_results in results.items():\n            if not benchmark_results:\n                summary[benchmark_name] = {\"total\": 0, \"passed\": 0, \"pass_rate\": 0.0, \"avg_score\": 0.0, \"avg_time\": 0.0}\n                continue\n            \n            total_tasks = len(benchmark_results)\n            passed_tasks = sum(1 for r in benchmark_results if r.passed)\n            avg_score = sum(r.score for r in benchmark_results) / total_tasks\n            avg_time = sum(r.execution_time for r in benchmark_results) / total_tasks\n            \n            summary[benchmark_name] = {\n                \"total\": total_tasks,\n                \"passed\": passed_tasks,\n                \"pass_rate\": passed_tasks / total_tasks if total_tasks > 0 else 0.0,\n                \"avg_score\": avg_score,\n                \"avg_time\": avg_time\n            }\n        \n        return summary\n\n\n# Aliases for compatibility\nPublicBenchmarkOrchestrator = FixedBenchmarkOrchestrator\nBenchmarkOrchestrator = FixedBenchmarkOrchestrator\n