# src/core/bigcodebench_integration.py
import subprocess
import json
import tempfile
import os
import sys
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
from pathlib import Path
import logging
from dataclasses import dataclass
import asyncio
import time

from .model_interfaces import ModelInterface, GenerationResult
from .custom_datasets import Task, TaskType, DifficultyLevel


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


class BigCodeBenchConfig(BaseModel):
    """Configuration for BigCodeBench integration"""
    bigcodebench_path: Optional[str] = None
    subset: str = "complete"  # complete, instruct, hard
    split: str = "test"
    max_new_tokens: int = 1024
    temperature: float = 0.1
    timeout: int = 300
    n_samples: int = 1
    resume: bool = True
    verbosity: int = 1


class BigCodeBenchRunner:
    """Runner for BigCodeBench evaluations"""
    
    def __init__(self, config: BigCodeBenchConfig = None):
        self.config = config or BigCodeBenchConfig()
        self.bigcodebench_available = self._check_bigcodebench_availability()
        
    def _check_bigcodebench_availability(self) -> bool:
        """Check if BigCodeBench is available"""
        try:
            result = subprocess.run(
                ["python", "-c", "import bigcodebench; print('available')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and "available" in result.stdout
        except Exception as e:
            logger.warning(f"BigCodeBench not available: {e}")
            return False
    
    def evaluate_with_bigcodebench(
        self, 
        model_interface: ModelInterface,
        subset: str = None,
        n_samples: int = None
    ) -> List[BigCodeBenchResult]:
        """Run evaluation using BigCodeBench"""
        if not self.bigcodebench_available:
            raise RuntimeError("BigCodeBench is not available. Please install with: pip install bigcodebench")
        
        subset = subset or self.config.subset
        n_samples = n_samples or self.config.n_samples
        
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate code for BigCodeBench tasks
            generated_file = self._generate_code_for_bigcodebench(
                model_interface, temp_dir, subset, n_samples
            )
            
            # Evaluate the generated code
            results = self._evaluate_generated_code(generated_file, temp_dir)
            
        return results
    
    def _generate_code_for_bigcodebench(
        self,
        model_interface: ModelInterface,
        temp_dir: str,
        subset: str,
        n_samples: int
    ) -> str:
        """Generate code for BigCodeBench tasks"""
        try:
            # Create custom generation script
            generation_script = self._create_generation_script(
                model_interface, temp_dir, subset, n_samples
            )
            
            # Run generation
            result = subprocess.run(
                ["python", generation_script],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout * n_samples
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Code generation failed: {result.stderr}")
            
            # Find generated file
            generated_files = list(Path(temp_dir).glob("*.jsonl"))
            if not generated_files:
                raise RuntimeError("No generated code file found")
            
            return str(generated_files[0])
            
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            raise
    
    def _create_generation_script(
        self,
        model_interface: ModelInterface,
        temp_dir: str,
        subset: str,
        n_samples: int
    ) -> str:
        """Create a script for code generation"""
        script_content = f"""
import json
import sys
from bigcodebench.data import get_bigcodebench, write_jsonl

# Add our model interface to path
sys.path.append('{os.path.dirname(os.path.abspath(__file__))}/../..')
from src.core.model_interfaces import ModelInterface, ModelConfig, ModelFactory

# Model configuration
model_config = {{
    'name': '{model_interface.name}',
    'provider': '{model_interface.provider}',
    'model_name': '{model_interface.model_name}',
    'base_url': '{getattr(model_interface.config, "base_url", "")}',
    'api_key': '{getattr(model_interface.config, "api_key", "")}',
    'temperature': {model_interface.config.temperature},
    'max_tokens': {model_interface.config.max_tokens}
}}

def generate_code():
    # Load BigCodeBench dataset
    problems = get_bigcodebench(subset="{subset}")
    
    # Initialize model interface
    interface = ModelFactory.create_interface(ModelConfig(**model_config))
    
    results = []
    for task_id, problem in problems.items():
        prompt = problem.get('prompt', '')
        
        # Generate code
        result = interface.generate_code(prompt)
        
        results.append({{
            'task_id': task_id,
            'completion': result.code,
            'generation_time': result.generation_time,
            'model': '{model_interface.model_name}'
        }})
        
        print(f"Generated code for {{task_id}}")
    
    # Save results
    output_file = "{temp_dir}/generated_code.jsonl"
    write_jsonl(output_file, results)
    print(f"Saved {{len(results)}} results to {{output_file}}")

if __name__ == "__main__":
    generate_code()
"""
        
        script_path = os.path.join(temp_dir, "generate_code.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def _evaluate_generated_code(self, generated_file: str, temp_dir: str) -> List[BigCodeBenchResult]:
        """Evaluate generated code using BigCodeBench"""
        try:
            # Run BigCodeBench evaluation
            eval_cmd = [
                "python", "-m", "bigcodebench.evaluate",
                "--samples", generated_file,
                "--split", self.config.split,
                "--subset", self.config.subset
            ]
            
            result = subprocess.run(
                eval_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout * 2
            )
            
            if result.returncode != 0:
                logger.warning(f"BigCodeBench evaluation had issues: {result.stderr}")
            
            # Parse results
            return self._parse_evaluation_results(temp_dir, generated_file)
            
        except Exception as e:
            logger.error(f"Failed to evaluate code: {e}")
            return []
    
    def _parse_evaluation_results(
        self, 
        temp_dir: str, 
        generated_file: str
    ) -> List[BigCodeBenchResult]:
        """Parse BigCodeBench evaluation results"""
        results = []
        
        try:
            # Look for result files
            result_files = list(Path(temp_dir).glob("*_eval_results.json"))
            
            if result_files:
                with open(result_files[0], 'r') as f:
                    eval_data = json.load(f)
                
                # Parse generated code
                generated_data = {}
                with open(generated_file, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        generated_data[item['task_id']] = item
                
                # Combine results
                for task_id, result in eval_data.get('eval', {}).items():
                    generated_item = generated_data.get(task_id, {})
                    
                    results.append(BigCodeBenchResult(
                        task_id=task_id,
                        model_name=generated_item.get('model', 'unknown'),
                        passed=result.get('passed', False),
                        score=float(result.get('score', 0.0)),
                        execution_time=generated_item.get('generation_time', 0.0),
                        generated_code=generated_item.get('completion', ''),
                        test_results=result
                    ))
            
            else:
                # Fallback: create basic results from generated file
                with open(generated_file, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        results.append(BigCodeBenchResult(
                            task_id=item['task_id'],
                            model_name=item.get('model', 'unknown'),
                            passed=False,  # Unknown without evaluation
                            score=0.0,
                            execution_time=item.get('generation_time', 0.0),
                            generated_code=item.get('completion', ''),
                            error_message="Evaluation results not available"
                        ))
        
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
        
        return results


class CustomBigCodeBenchRunner(BigCodeBenchRunner):
    """Extended runner that integrates custom datasets with BigCodeBench"""
    
    def __init__(self, config: BigCodeBenchConfig = None):
        super().__init__(config)
        
    def evaluate_custom_tasks(
        self,
        model_interface: ModelInterface,
        tasks: List[Task]
    ) -> List[BigCodeBenchResult]:
        """Evaluate custom tasks using BigCodeBench-style evaluation"""
        results = []
        
        for task in tasks:
            try:
                start_time = time.time()
                
                # Generate code
                generation_result = model_interface.generate_code(
                    prompt=task.prompt,
                    system_prompt=task.system_prompt
                )
                
                # Evaluate the generated code
                evaluation_result = self._evaluate_custom_task(task, generation_result)
                
                results.append(BigCodeBenchResult(
                    task_id=task.task_id,
                    model_name=model_interface.model_name,
                    passed=evaluation_result['passed'],
                    score=evaluation_result['score'],
                    execution_time=generation_result.generation_time,
                    generated_code=generation_result.code,
                    test_results=evaluation_result,
                    error_message=generation_result.error
                ))
                
            except Exception as e:
                logger.error(f"Failed to evaluate task {task.task_id}: {e}")
                results.append(BigCodeBenchResult(
                    task_id=task.task_id,
                    model_name=model_interface.model_name,
                    passed=False,
                    score=0.0,
                    execution_time=0.0,
                    generated_code="",
                    error_message=str(e)
                ))
        
        return results
    
    def _evaluate_custom_task(self, task: Task, generation_result: GenerationResult) -> Dict:
        """Evaluate a custom task"""
        evaluation = {
            'passed': False,
            'score': 0.0,
            'details': {},
            'test_results': []
        }
        
        if generation_result.error:
            evaluation['details']['error'] = generation_result.error
            return evaluation
        
        code = generation_result.code
        
        # Basic evaluation metrics
        scores = {}
        
        # 1. Code quality check
        scores['code_quality'] = self._evaluate_code_quality(code, task)
        
        # 2. Functionality check (if test cases exist)
        if task.test_cases:
            scores['functionality'] = self._evaluate_functionality(code, task)
        else:
            scores['functionality'] = self._evaluate_basic_functionality(code, task)
        
        # 3. Security check
        scores['security'] = self._evaluate_security(code, task)
        
        # 4. Performance check
        scores['performance'] = self._evaluate_performance(code, task)
        
        # 5. Task-specific evaluation
        if task.task_type == TaskType.FRONTEND:
            scores['accessibility'] = self._evaluate_accessibility(code, task)
        elif task.task_type == TaskType.BACKEND:
            scores['api_design'] = self._evaluate_api_design(code, task)
        elif task.task_type == TaskType.TESTING:
            scores['test_coverage'] = self._evaluate_test_coverage(code, task)
        
        # Calculate weighted score
        criteria = task.evaluation_criteria
        total_score = (
            scores['functionality'] * criteria.functionality +
            scores['code_quality'] * criteria.code_quality +
            scores['security'] * criteria.security +
            scores['performance'] * criteria.performance
        )
        
        # Add task-specific weight
        if task.task_type == TaskType.FRONTEND and 'accessibility' in scores:
            total_score += scores['accessibility'] * criteria.accessibility
        
        evaluation['score'] = min(total_score, 1.0)
        evaluation['passed'] = evaluation['score'] >= 0.7  # 70% threshold
        evaluation['details'] = scores
        
        return evaluation
    
    def _evaluate_code_quality(self, code: str, task: Task) -> float:
        """Evaluate code quality"""
        score = 0.8  # Base score
        
        # Check for common quality indicators
        quality_checks = [
            ('comments' in code.lower(), 0.1),
            ('class ' in code or 'def ' in code, 0.1),
            (len(code.strip()) > 50, 0.1),  # Substantial code
            ('import ' in code, 0.1),  # Uses imports
            ('return ' in code, 0.1),  # Has returns
        ]
        
        for check, weight in quality_checks:
            if check:
                score += weight
        
        return min(score, 1.0)
    
    def _evaluate_functionality(self, code: str, task: Task) -> float:
        """Evaluate functionality using test cases"""
        if not task.test_cases:
            return self._evaluate_basic_functionality(code, task)
        
        # For now, do basic heuristic checking
        # In a full implementation, you'd execute the test cases
        score = 0.0
        
        for test_case in task.test_cases:
            # Simple heuristic: check if code contains expected elements
            if self._contains_expected_elements(code, task):
                score += 1.0 / len(task.test_cases)
        
        return min(score, 1.0)
    
    def _evaluate_basic_functionality(self, code: str, task: Task) -> float:
        """Basic functionality evaluation without test cases"""
        score = 0.5  # Base score
        
        # Check for task-specific requirements
        expected_tech = [tech.lower() for tech in task.expected_technologies]
        
        functionality_checks = []
        
        if 'react' in expected_tech:
            functionality_checks.extend([
                ('function ' in code.lower() or 'const ' in code.lower(), 0.2),
                ('return' in code.lower(), 0.2),
                ('props' in code.lower() or 'useState' in code, 0.1)
            ])
        
        if 'python' in expected_tech or task.task_type == TaskType.BACKEND:
            functionality_checks.extend([
                ('def ' in code, 0.2),
                ('class ' in code, 0.1),
                ('return ' in code, 0.2)
            ])
        
        if task.task_type == TaskType.TESTING:
            functionality_checks.extend([
                ('test' in code.lower(), 0.3),
                ('assert' in code.lower(), 0.2)
            ])
        
        for check, weight in functionality_checks:
            if check:
                score += weight
        
        return min(score, 1.0)
    
    def _contains_expected_elements(self, code: str, task: Task) -> bool:
        """Check if code contains expected elements"""
        code_lower = code.lower()
        
        # Basic checks based on task type
        if task.task_type == TaskType.FRONTEND:
            return any(word in code_lower for word in ['component', 'function', 'const', 'return'])
        elif task.task_type == TaskType.BACKEND:
            return any(word in code_lower for word in ['def', 'class', 'api', 'route'])
        elif task.task_type == TaskType.TESTING:
            return any(word in code_lower for word in ['test', 'assert', 'expect'])
        
        return True
    
    def _evaluate_security(self, code: str, task: Task) -> float:
        """Evaluate security considerations"""
        score = 0.8  # Base score
        
        # Check for security anti-patterns
        security_issues = [
            'eval(' in code,
            'exec(' in code,
            'input(' in code and task.task_type == TaskType.BACKEND,
            'os.system(' in code,
            'subprocess.call(' in code.replace(' ', ''),
        ]
        
        # Deduct for security issues
        for issue in security_issues:
            if issue:
                score -= 0.2
        
        return max(score, 0.0)
    
    def _evaluate_performance(self, code: str, task: Task) -> float:
        """Evaluate performance considerations"""
        score = 0.7  # Base score
        
        # Check for performance-related patterns
        perf_indicators = [
            ('async' in code and task.task_type != TaskType.TESTING, 0.1),
            ('cache' in code.lower(), 0.1),
            ('index' in code.lower() and task.task_type == TaskType.BACKEND, 0.1),
            (len(code.split('\n')) < 100, 0.1)  # Concise code
        ]
        
        for check, weight in perf_indicators:
            if check:
                score += weight
        
        return min(score, 1.0)
    
    def _evaluate_accessibility(self, code: str, task: Task) -> float:
        """Evaluate accessibility for frontend code"""
        score = 0.6  # Base score
        
        accessibility_checks = [
            ('aria-' in code.lower(), 0.2),
            ('role=' in code.lower(), 0.1),
            ('alt=' in code.lower(), 0.1),
            ('tabindex' in code.lower(), 0.1)
        ]
        
        for check, weight in accessibility_checks:
            if check:
                score += weight
        
        return min(score, 1.0)
    
    def _evaluate_api_design(self, code: str, task: Task) -> float:
        """Evaluate API design for backend code"""
        score = 0.6  # Base score
        
        api_checks = [
            ('get' in code.lower() and 'post' in code.lower(), 0.2),
            ('status_code' in code.lower() or 'response' in code.lower(), 0.1),
            ('json' in code.lower(), 0.1),
            ('validate' in code.lower() or 'schema' in code.lower(), 0.1)
        ]
        
        for check, weight in api_checks:
            if check:
                score += weight
        
        return min(score, 1.0)
    
    def _evaluate_test_coverage(self, code: str, task: Task) -> float:
        """Evaluate test coverage for testing code"""
        score = 0.6  # Base score
        
        test_checks = [
            (code.count('def test_') >= 3, 0.2),
            ('setUp' in code or 'fixture' in code.lower(), 0.1),
            ('mock' in code.lower(), 0.1),
            ('assert' in code.lower(), 0.1)
        ]
        
        for check, weight in test_checks:
            if check:
                score += weight
        
        return min(score, 1.0)


# Example usage
if __name__ == "__main__":
    from .model_interfaces import ModelConfig, ModelFactory
    from .custom_datasets import DatasetManager, TaskType
    
    # Test BigCodeBench integration
    config = ModelConfig(
        name="Test Model",
        provider="ollama",
        model_name="codellama:7b"
    )
    
    interface = ModelFactory.create_interface(config)
    runner = CustomBigCodeBenchRunner()
    
    # Test with custom tasks
    dataset_manager = DatasetManager()
    frontend_tasks = dataset_manager.get_tasks_by_type(TaskType.FRONTEND)[:1]  # Test one task
    
    results = runner.evaluate_custom_tasks(interface, frontend_tasks)
    
    for result in results:
        print(f"Task: {result.task_id}")
        print(f"Score: {result.score:.2f}")
        print(f"Passed: {result.passed}")
        print(f"Time: {result.execution_time:.2f}s")
        print("---")