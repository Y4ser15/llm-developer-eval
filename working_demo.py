#!/usr/bin/env python3
"""
Simple Working Demo - LLM Coding Evaluation Platform
This script demonstrates the platform working end-to-end with a mock model.
"""

import sys
import asyncio
import time
from pathlib import Path
from typing import Optional

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import our components
from src.core.model_interfaces import ModelInterface, ModelConfig, GenerationResult
from src.core.custom_datasets import DatasetManager, TaskType, DifficultyLevel
from src.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig
from src.core.bigcodebench_integration import CustomBigCodeBenchRunner


class MockModelInterface(ModelInterface):
    """Mock model interface for demonstration purposes"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
    def test_connection(self) -> bool:
        """Mock connection test - always passes"""
        return True
    
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate mock code based on task type"""
        start_time = time.time()
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Generate different code based on prompt content
        if "react" in prompt.lower() or "component" in prompt.lower():
            # Frontend React code
            code = """import React from 'react';

const UserProfileCard = ({ user }) => {
  if (!user) return null;
  
  return (
    <div className="user-profile-card">
      <div className="avatar">
        {user.avatar ? (
          <img src={user.avatar} alt={user.name} />
        ) : (
          <div className="avatar-placeholder">
            {user.name?.charAt(0)?.toUpperCase()}
          </div>
        )}
      </div>
      <div className="user-info">
        <h3>{user.name}</h3>
        <p>{user.email}</p>
        <span className="role">{user.role}</span>
      </div>
    </div>
  );
};

export default UserProfileCard;"""

        elif "api" in prompt.lower() or "backend" in prompt.lower():
            # Backend API code
            code = """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class User(BaseModel):
    id: Optional[int] = None
    username: str
    email: str
    created_at: Optional[str] = None

users_db = []

@app.post("/users", response_model=User)
async def create_user(user: User):
    user.id = len(users_db) + 1
    users_db.append(user)
    return user

@app.get("/users", response_model=List[User])
async def get_users():
    return users_db

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    user = next((u for u in users_db if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user"""

        elif "test" in prompt.lower():
            # Testing code
            code = """import pytest
from unittest.mock import Mock

def calculate_discount(price, discount_percent, customer_type='regular'):
    if price <= 0:
        raise ValueError("Price must be positive")
    
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    
    discounted_price = price * (1 - discount_percent / 100)
    
    if customer_type == 'premium':
        discounted_price *= 0.95
    elif customer_type == 'vip':
        discounted_price *= 0.9
    
    return round(discounted_price, 2)

def test_calculate_discount_valid_inputs():
    assert calculate_discount(100, 10) == 90.0
    assert calculate_discount(100, 0) == 100.0
    assert calculate_discount(100, 100) == 0.0

def test_calculate_discount_customer_types():
    assert calculate_discount(100, 10, 'regular') == 90.0
    assert calculate_discount(100, 10, 'premium') == 85.5
    assert calculate_discount(100, 10, 'vip') == 81.0

def test_calculate_discount_edge_cases():
    with pytest.raises(ValueError):
        calculate_discount(-10, 10)
    
    with pytest.raises(ValueError):
        calculate_discount(100, 150)"""

        else:
            # Generic code
            code = """# Generated code for the given task
def solve_problem():
    # Implementation would go here
    pass

if __name__ == "__main__":
    solve_problem()"""
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            code=code,
            model_name=self.model_name,
            provider=self.provider,
            generation_time=generation_time,
            token_count=len(code.split())
        )
    
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version - just calls sync version for mock"""
        return self.generate_code(prompt, system_prompt)


class SimpleDemoRunner:
    """Simple demo runner that shows the platform working"""
    
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.evaluation_engine = EvaluationEngine()
        
    async def run_demo(self):
        """Run a simple evaluation demo"""
        print("🚀 LLM Coding Evaluation Platform - Working Demo")
        print("=" * 60)
        
        # 1. Show dataset information
        print("\n📊 Dataset Information:")
        summary = self.dataset_manager.export_tasks_summary()
        print(f"   Total tasks available: {summary['total_tasks']}")
        print(f"   Frontend tasks: {summary['by_type']['frontend']}")
        print(f"   Backend tasks: {summary['by_type']['backend']}")
        print(f"   Testing tasks: {summary['by_type']['testing']}")
        
        # 2. Get sample tasks
        print("\n📋 Sample Tasks:")
        frontend_tasks = self.dataset_manager.get_tasks_by_type(TaskType.FRONTEND)
        backend_tasks = self.dataset_manager.get_tasks_by_type(TaskType.BACKEND)
        testing_tasks = self.dataset_manager.get_tasks_by_type(TaskType.TESTING)
        
        sample_tasks = []
        if frontend_tasks:
            sample_tasks.append(frontend_tasks[0])  # Easy React task
        if backend_tasks:
            sample_tasks.append(backend_tasks[0])   # Easy API task  
        if testing_tasks:
            sample_tasks.append(testing_tasks[0])   # Easy testing task
        
        for task in sample_tasks:
            print(f"   - {task.title} ({task.task_type.value}, {task.difficulty.value})")
        
        # 3. Create mock model configurations
        print("\n🤖 Mock Model Configurations:")
        mock_models = [
            ModelConfig(
                name="MockLlama-7B",
                provider="mock",
                model_name="mock-codellama-7b"
            ),
            ModelConfig(
                name="MockGPT-4",
                provider="mock", 
                model_name="mock-gpt-4"
            )
        ]
        
        for config in mock_models:
            print(f"   - {config.name} ({config.provider})")
        
        # 4. Run evaluation with mock models
        print("\n🔄 Running Evaluation...")
        
        # Create evaluation config
        eval_config = EvaluationConfig(
            task_types=[TaskType.FRONTEND, TaskType.BACKEND, TaskType.TESTING],
            difficulty_levels=[DifficultyLevel.EASY],
            max_tasks_per_type=1,  # Just one task per type for demo
            include_bigcodebench=False,  # Skip BigCodeBench for demo
            parallel_execution=False,
            generate_detailed_report=False,
            export_results=True
        )
        
        print(f"   Task types: {[t.value for t in eval_config.task_types]}")
        print(f"   Difficulty: {[d.value for d in eval_config.difficulty_levels]}")
        print(f"   Max tasks per type: {eval_config.max_tasks_per_type}")
        
        # 5. Create mock model interfaces
        mock_interfaces = []
        for config in mock_models:
            interface = MockModelInterface(config)
            mock_interfaces.append(interface)
        
        # 6. Run evaluation
        def progress_callback(message):
            print(f"   📈 {message}")
        
        print("\n⚡ Starting evaluation...")
        start_time = time.time()
        
        results = await self.run_mock_evaluation(mock_interfaces, eval_config, progress_callback)
        
        eval_time = time.time() - start_time
        print(f"\n✅ Evaluation completed in {eval_time:.2f} seconds")
        
        # 7. Display results
        print("\n📊 Results Summary:")
        leaderboard = results.get_leaderboard()
        
        print("   🏆 Leaderboard:")
        for i, entry in enumerate(leaderboard, 1):
            print(f"      {i}. {entry['model_name']}")
            print(f"         Overall Score: {entry['overall_score']:.3f}")
            print(f"         Pass Rate: {entry['pass_rate']:.1%}")
            print(f"         Avg Time: {entry['avg_time']:.2f}s")
            print(f"         Frontend: {entry['frontend_score']:.3f}")
            print(f"         Backend: {entry['backend_score']:.3f}")
            print(f"         Testing: {entry['testing_score']:.3f}")
            print()
        
        # 8. Show detailed results for one model
        if results.model_results:
            model_result = results.model_results[0]
            print(f"🔍 Detailed Results for {model_result.model_name}:")
            
            for task_result in model_result.task_results[:3]:  # Show first 3
                print(f"   📝 {task_result.task_title}")
                print(f"      Score: {task_result.score:.3f}")
                print(f"      Passed: {'✅' if task_result.passed else '❌'}")
                print(f"      Time: {task_result.execution_time:.2f}s")
                print(f"      Code length: {len(task_result.generated_code)} chars")
                print()
        
        print("🎉 Demo completed successfully!")
        print("\n🚀 Next Steps:")
        print("   1. Replace mock models with real LLM interfaces")
        print("   2. Add BigCodeBench integration for more comprehensive evaluation")
        print("   3. Start the web interface: python setup.py --mode start")
        print("   4. Configure real models and run full evaluations")
        
        return results
    
    async def run_mock_evaluation(self, mock_interfaces, config, progress_callback):
        """Run evaluation with mock model interfaces"""
        from src.evaluation.evaluation_engine import EvaluationRun, ModelResult, TaskResult
        from datetime import datetime
        import uuid
        
        # Create evaluation run
        evaluation_run = EvaluationRun(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            config=config,
            status="running"
        )
        
        start_time = time.time()
        
        # Get tasks to evaluate
        tasks = self.evaluation_engine._select_tasks(config)
        progress_callback(f"Selected {len(tasks)} tasks for evaluation")
        
        # Evaluate each mock model
        for i, interface in enumerate(mock_interfaces):
            progress_callback(f"Evaluating model {i+1}/{len(mock_interfaces)}: {interface.name}")
            
            model_result = ModelResult(
                model_name=interface.model_name,
                provider=interface.provider,
                model_config=interface.config.dict()
            )
            
            # Evaluate each task
            for j, task in enumerate(tasks):
                progress_callback(f"Task {j+1}/{len(tasks)}: {task.title}")
                
                # Generate code using mock interface
                generation_result = interface.generate_code(
                    prompt=task.prompt,
                    system_prompt=task.system_prompt
                )
                
                # Simple scoring based on code length and content
                score = self.calculate_mock_score(generation_result.code, task)
                passed = score >= 0.7
                
                task_result = TaskResult(
                    task_id=task.task_id,
                    task_title=task.title,
                    task_type=task.task_type,
                    difficulty=task.difficulty,
                    model_name=interface.model_name,
                    provider=interface.provider,
                    score=score,
                    passed=passed,
                    execution_time=generation_result.generation_time,
                    generated_code=generation_result.code
                )
                
                model_result.task_results.append(task_result)
            
            # Calculate summary
            model_result.calculate_summary()
            evaluation_run.model_results.append(model_result)
        
        evaluation_run.duration = time.time() - start_time
        evaluation_run.status = "completed"
        
        return evaluation_run
    
    def calculate_mock_score(self, code: str, task) -> float:
        """Calculate a mock score based on code quality indicators"""
        score = 0.5  # Base score
        
        # Basic quality checks
        if len(code) > 100:  # Substantial code
            score += 0.1
        
        if "import" in code or "from" in code:  # Uses imports
            score += 0.1
        
        if "def " in code or "function" in code or "class" in code:  # Has functions/classes
            score += 0.1
        
        if "return" in code:  # Has returns
            score += 0.1
        
        # Task-specific checks
        if task.task_type == TaskType.FRONTEND:
            if "react" in code.lower() or "component" in code.lower():
                score += 0.2
            if "export" in code.lower():
                score += 0.1
        
        elif task.task_type == TaskType.BACKEND:
            if "api" in code.lower() or "app" in code.lower():
                score += 0.2
            if "post" in code.lower() or "get" in code.lower():
                score += 0.1
        
        elif task.task_type == TaskType.TESTING:
            if "test" in code.lower():
                score += 0.2
            if "assert" in code.lower():
                score += 0.1
        
        return min(score, 1.0)


async def main():
    """Main demo function"""
    demo = SimpleDemoRunner()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
