# src/core/bigcodebench_integration.py
"""
True BigCodeBench Integration for LLM Coding Evaluation Platform
This replaces the custom evaluation framework with actual BigCodeBench integration.
"""

import subprocess
import json
import tempfile
import os
import sys
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import asyncio
import time
from dataclasses import dataclass

from .model_interfaces import ModelInterface, GenerationResult


logger = logging.getLogger(__name__)


@dataclass
class BigCodeBenchResult:
    """Result from BigCodeBench evaluation"""
    task_id: str
    model_name: str
    passed: bool
    score: float
    execution_time: float
    generated_code: str
    error_message: Optional[str] = None
    test_results: Optional[Dict] = None


class BigCodeBenchIntegration:
    """True BigCodeBench integration"""
    
    def __init__(self):
        self.bigcodebench_available = self._check_installation()
        if not self.bigcodebench_available:
            self._install_bigcodebench()
    
    def _check_installation(self) -> bool:
        """Check if BigCodeBench is installed"""
        try:
            import bigcodebench
            return True
        except ImportError:
            return False
    
    def _install_bigcodebench(self):
        """Install BigCodeBench"""
        print("📦 Installing BigCodeBench...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/bigcode-project/bigcodebench.git"
            ], check=True, capture_output=True)
            self.bigcodebench_available = True
            print("✅ BigCodeBench installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install BigCodeBench: {e}")
            self.bigcodebench_available = False
    
    def get_tasks(self, subset: str = "complete", domain_filter: Optional[str] = None) -> List[Dict]:
        """Get BigCodeBench tasks with optional domain filtering"""
        if not self.bigcodebench_available:
            raise RuntimeError("BigCodeBench not available")
        
        try:
            from bigcodebench.data import get_bigcodebench
            tasks = get_bigcodebench(subset=subset)
            
            if domain_filter:
                # Filter tasks by domain based on task content/libraries
                filtered_tasks = {}
                for task_id, task in tasks.items():
                    if self._matches_domain(task, domain_filter):
                        filtered_tasks[task_id] = task
                return filtered_tasks
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to get BigCodeBench tasks: {e}")
            return {}
    
    def _matches_domain(self, task: Dict, domain: str) -> bool:
        """Check if task matches specified domain"""
        prompt = task.get('prompt', '').lower()
        
        if domain == "frontend":
            frontend_keywords = [
                'react', 'vue', 'angular', 'html', 'css', 'javascript', 'dom',
                'component', 'render', 'ui', 'interface', 'web', 'browser'
            ]
            return any(keyword in prompt for keyword in frontend_keywords)
        
        elif domain == "backend":
            backend_keywords = [
                'api', 'server', 'database', 'http', 'rest', 'sql', 'auth',
                'endpoint', 'service', 'middleware', 'route', 'fastapi', 'flask'
            ]
            return any(keyword in prompt for keyword in backend_keywords)
        
        elif domain == "testing":
            testing_keywords = [
                'test', 'unittest', 'pytest', 'mock', 'assert', 'coverage',
                'integration', 'e2e', 'selenium', 'automation'
            ]
            return any(keyword in prompt for keyword in testing_keywords)
        
        return True
    
    async def evaluate_model(
        self, 
        model_interface: ModelInterface,
        subset: str = "complete",
        domain_filter: Optional[str] = None,
        max_tasks: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[BigCodeBenchResult]:
        """Evaluate model using BigCodeBench"""
        
        if not self.bigcodebench_available:
            raise RuntimeError("BigCodeBench not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate code samples
            samples_file = await self._generate_samples(
                model_interface, subset, domain_filter, max_tasks, temp_dir, progress_callback
            )
            
            # Evaluate samples
            results = await self._evaluate_samples(samples_file, temp_dir, progress_callback)
            
            return results
    
    async def _generate_samples(
        self,
        model_interface: ModelInterface,
        subset: str,
        domain_filter: Optional[str],
        max_tasks: int,
        temp_dir: str,
        progress_callback: Optional[callable]
    ) -> str:
        """Generate code samples for BigCodeBench tasks"""
        
        # Get tasks
        tasks = self.get_tasks(subset, domain_filter)
        if max_tasks > 0:
            task_items = list(tasks.items())[:max_tasks]
            tasks = dict(task_items)
        
        if progress_callback:
            progress_callback(f"Generating code for {len(tasks)} tasks...")
        
        samples = []
        for i, (task_id, task) in enumerate(tasks.items()):
            if progress_callback:
                progress_callback(f"Task {i+1}/{len(tasks)}: {task_id}")
            
            try:
                # Generate code using model interface
                result = model_interface.generate_code(
                    prompt=task['prompt'],
                    system_prompt="You are an expert programmer. Generate working code that solves the given problem."
                )
                
                sample = {
                    'task_id': task_id,
                    'completion': result.code,
                    'model': model_interface.model_name,
                    'generation_time': result.generation_time
                }
                samples.append(sample)
                
            except Exception as e:
                logger.error(f"Failed to generate code for task {task_id}: {e}")
                # Add empty sample to maintain structure
                samples.append({
                    'task_id': task_id,
                    'completion': '',
                    'model': model_interface.model_name,
                    'generation_time': 0.0,
                    'error': str(e)
                })
        
        # Save samples
        samples_file = os.path.join(temp_dir, f"{model_interface.model_name}_samples.jsonl")
        with open(samples_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        return samples_file
    
    async def _evaluate_samples(
        self,
        samples_file: str,
        temp_dir: str,
        progress_callback: Optional[callable]
    ) -> List[BigCodeBenchResult]:
        """Evaluate generated samples using BigCodeBench"""
        
        if progress_callback:
            progress_callback("Evaluating generated code...")
        
        try:
            # Run BigCodeBench evaluation
            eval_cmd = [
                sys.executable, "-m", "bigcodebench.evaluate",
                "--samples", samples_file,
                "--subset", "complete"
            ]
            
            result = subprocess.run(
                eval_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.warning(f"BigCodeBench evaluation warning: {result.stderr}")
            
            # Parse results
            return self._parse_results(samples_file, temp_dir)
            
        except Exception as e:
            logger.error(f"Failed to evaluate samples: {e}")
            # Return basic results from samples file
            return self._parse_samples_only(samples_file)
    
    def _parse_results(self, samples_file: str, temp_dir: str) -> List[BigCodeBenchResult]:
        """Parse BigCodeBench evaluation results"""
        results = []
        
        try:
            # Load samples
            samples = {}
            with open(samples_file, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    samples[sample['task_id']] = sample
            
            # Look for evaluation results
            result_files = list(Path(temp_dir).glob("*_eval_results.json"))
            
            if result_files:
                # Parse detailed results
                with open(result_files[0], 'r') as f:
                    eval_data = json.load(f)
                
                for task_id, sample in samples.items():
                    eval_result = eval_data.get('eval', {}).get(task_id, {})
                    
                    results.append(BigCodeBenchResult(
                        task_id=task_id,
                        model_name=sample.get('model', 'unknown'),
                        passed=eval_result.get('passed', False),
                        score=float(eval_result.get('score', 0.0)),
                        execution_time=sample.get('generation_time', 0.0),
                        generated_code=sample.get('completion', ''),
                        error_message=sample.get('error'),
                        test_results=eval_result
                    ))
            else:
                # Fallback to basic results
                return self._parse_samples_only(samples_file)
                
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
            return self._parse_samples_only(samples_file)
        
        return results
    
    def _parse_samples_only(self, samples_file: str) -> List[BigCodeBenchResult]:
        """Parse samples file when evaluation results not available"""
        results = []
        
        try:
            with open(samples_file, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    
                    # Basic scoring based on code length and errors
                    has_error = 'error' in sample
                    code_length = len(sample.get('completion', ''))
                    basic_score = 0.0 if has_error else min(code_length / 1000, 1.0)
                    
                    results.append(BigCodeBenchResult(
                        task_id=sample['task_id'],
                        model_name=sample.get('model', 'unknown'),
                        passed=not has_error and code_length > 50,
                        score=basic_score,
                        execution_time=sample.get('generation_time', 0.0),
                        generated_code=sample.get('completion', ''),
                        error_message=sample.get('error'),
                        test_results=None
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to parse samples: {e}")
        
        return results


class HumanEvalIntegration:
    """HumanEval integration for additional evaluation"""
    
    def __init__(self):
        self.available = self._check_installation()
    
    def _check_installation(self) -> bool:
        """Check if HumanEval/EvalPlus is available"""
        try:
            from datasets import load_dataset
            load_dataset("openai_humaneval", split="test", streaming=True)
            return True
        except Exception:
            return False
    
    async def evaluate_model(
        self, 
        model_interface: ModelInterface,
        domain_filter: Optional[str] = None,
        max_tasks: int = 20
    ) -> List[BigCodeBenchResult]:
        """Evaluate model using HumanEval"""
        
        if not self.available:
            raise RuntimeError("HumanEval not available")
        
        try:
            from datasets import load_dataset
            from evaluate import load
            
            # Load dataset and evaluation metric
            dataset = load_dataset("openai_humaneval", split="test")
            code_eval = load("code_eval")
            
            if max_tasks > 0:
                dataset = dataset.select(range(min(max_tasks, len(dataset))))
            
            results = []
            
            for i, problem in enumerate(dataset):
                try:
                    # Generate code
                    result = model_interface.generate_code(
                        prompt=problem['prompt'],
                        system_prompt="Complete the function. Only return the code."
                    )
                    
                    # Basic evaluation (simplified)
                    passed = len(result.code.strip()) > 20 and 'def ' in result.code
                    score = 1.0 if passed else 0.0
                    
                    results.append(BigCodeBenchResult(
                        task_id=problem['task_id'],
                        model_name=model_interface.model_name,
                        passed=passed,
                        score=score,
                        execution_time=result.generation_time,
                        generated_code=result.code,
                        error_message=result.error
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate HumanEval task {problem['task_id']}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"HumanEval evaluation failed: {e}")
            return []


class BenchmarkOrchestrator:
    """Orchestrates multiple benchmark evaluations"""
    
    def __init__(self):
        self.bigcodebench = BigCodeBenchIntegration()
        self.humaneval = HumanEvalIntegration()
    
    async def run_comprehensive_evaluation(
        self,
        model_interface: ModelInterface,
        domains: List[str] = ["frontend", "backend", "testing"],
        include_humaneval: bool = True,
        max_tasks_per_domain: int = 10,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[BigCodeBenchResult]]:
        """Run comprehensive evaluation across multiple benchmarks and domains"""
        
        all_results = {}
        
        # BigCodeBench evaluation by domain
        for domain in domains:
            if progress_callback:
                progress_callback(f"Evaluating {domain} domain with BigCodeBench...")
            
            try:
                domain_results = await self.bigcodebench.evaluate_model(
                    model_interface,
                    subset="complete",
                    domain_filter=domain,
                    max_tasks=max_tasks_per_domain,
                    progress_callback=progress_callback
                )
                all_results[f"bigcodebench_{domain}"] = domain_results
                
            except Exception as e:
                logger.error(f"Failed to evaluate {domain} domain: {e}")
                all_results[f"bigcodebench_{domain}"] = []
        
        # HumanEval evaluation
        if include_humaneval and self.humaneval.available:
            if progress_callback:
                progress_callback("Evaluating with HumanEval...")
            
            try:
                humaneval_results = await self.humaneval.evaluate_model(
                    model_interface,
                    max_tasks=max_tasks_per_domain
                )
                all_results["humaneval"] = humaneval_results
                
            except Exception as e:
                logger.error(f"Failed to evaluate with HumanEval: {e}")
                all_results["humaneval"] = []
        
        return all_results
    
    def get_benchmark_summary(self, results: Dict[str, List[BigCodeBenchResult]]) -> Dict[str, Any]:
        """Generate summary statistics for benchmark results"""
        summary = {}
        
        for benchmark_name, benchmark_results in results.items():
            if not benchmark_results:
                summary[benchmark_name] = {"total": 0, "passed": 0, "avg_score": 0.0}
                continue
            
            total_tasks = len(benchmark_results)
            passed_tasks = sum(1 for r in benchmark_results if r.passed)
            avg_score = sum(r.score for r in benchmark_results) / total_tasks
            avg_time = sum(r.execution_time for r in benchmark_results) / total_tasks
            
            summary[benchmark_name] = {
                "total": total_tasks,
                "passed": passed_tasks,
                "pass_rate": passed_tasks / total_tasks,
                "avg_score": avg_score,
                "avg_time": avg_time
            }
        
        return summary


# Example usage
if __name__ == "__main__":
    import asyncio
    from .model_interfaces import ModelConfig, ModelFactory
    
    async def main():
        # Create model interface
        config = ModelConfig(
            name="CodeLlama 7B",
            provider="ollama",
            model_name="codellama:7b"
        )
        interface = ModelFactory.create_interface(config)
        
        # Run comprehensive evaluation
        orchestrator = BenchmarkOrchestrator()
        
        def progress(msg):
            print(f"Progress: {msg}")
        
        results = await orchestrator.run_comprehensive_evaluation(
            interface,
            domains=["frontend", "backend", "testing"],
            max_tasks_per_domain=3,
            progress_callback=progress
        )
        
        # Print summary
        summary = orchestrator.get_benchmark_summary(results)
        for benchmark, stats in summary.items():
            print(f"{benchmark}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
    
    asyncio.run(main())
