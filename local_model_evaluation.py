#!/usr/bin/env python3
"""
Real Model Evaluation - LLM Coding Evaluation Platform
This script runs actual evaluation with real Ollama models.
"""

import sys
import asyncio
import requests
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.core.model_interfaces import ModelConfig, ModelFactory
from src.core.custom_datasets import DatasetManager, TaskType, DifficultyLevel
from src.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig


class RealModelEvaluator:
    """Evaluator for real Ollama models"""
    
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.evaluation_engine = EvaluationEngine()
        self.ollama_base_url = "http://localhost:11434"
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception:
            return []
    
    def get_recommended_models(self) -> list:
        """Get recommended coding models"""
        recommended = [
            "codellama:7b",
            "codellama:13b", 
            "deepseek-coder:6.7b",
            "deepseek-coder:33b",
            "qwen2.5-coder:7b",
            "qwen2.5-coder:32b",
            "starcoder2:7b",
            "codegemma:7b"
        ]
        
        available = self.get_available_models()
        return [model for model in recommended if model in available]
    
    async def run_real_evaluation(self, selected_models: list = None, quick_mode: bool = True):
        """Run evaluation with real models"""
        print("🚀 LLM Coding Evaluation Platform - Real Model Evaluation")
        print("=" * 70)
        
        # 1. Check Ollama connection
        print("🔍 Checking Ollama connection...")
        if not self.check_ollama_connection():
            print("❌ Ollama is not running!")
            print("💡 Please start Ollama with: ollama serve")
            print("📥 Download models with: ollama pull <model_name>")
            return
        
        print("✅ Ollama is running")
        
        # 2. Get available models
        available_models = self.get_available_models()
        recommended = self.get_recommended_models()
        
        print(f"📦 Available models: {len(available_models)}")
        print(f"🎯 Recommended coding models: {len(recommended)}")
        
        if not available_models:
            print("❌ No models found!")
            print("📥 Download a model with: ollama pull codellama:7b")
            return
        
        # 3. Select models to evaluate
        if selected_models:
            models_to_test = [m for m in selected_models if m in available_models]
        else:
            models_to_test = recommended[:2] if recommended else available_models[:2]
        
        if not models_to_test:
            print("❌ No suitable models found for testing")
            return
        
        print(f"🧪 Testing models: {models_to_test}")
        
        # 4. Create model configurations
        model_configs = []
        for model_name in models_to_test:
            config = ModelConfig(
                name=f"Ollama-{model_name}",
                provider="ollama",
                model_name=model_name,
                base_url=self.ollama_base_url,
                temperature=0.1,
                max_tokens=2048
            )
            model_configs.append(config)
        
        # 5. Configure evaluation
        if quick_mode:
            eval_config = EvaluationConfig(
                task_types=[TaskType.FRONTEND, TaskType.BACKEND, TaskType.TESTING],
                difficulty_levels=[DifficultyLevel.EASY],
                max_tasks_per_type=1,  # Quick test - 1 task per type
                include_bigcodebench=False,
                parallel_execution=False,
                generate_detailed_report=True,
                export_results=True
            )
            print("⚡ Quick mode: 1 easy task per type")
        else:
            eval_config = EvaluationConfig(
                task_types=[TaskType.FRONTEND, TaskType.BACKEND, TaskType.TESTING],
                difficulty_levels=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM],
                max_tasks_per_type=2,
                include_bigcodebench=False,
                parallel_execution=False,
                generate_detailed_report=True,
                export_results=True
            )
            print("🔄 Full mode: 2 easy + medium tasks per type")
        
        # 6. Show what will be evaluated
        tasks = self.evaluation_engine._select_tasks(eval_config)
        print(f"📝 Tasks to evaluate: {len(tasks)}")
        for task in tasks:
            print(f"   - {task.title} ({task.task_type.value}, {task.difficulty.value})")
        
        # 7. Estimate time
        estimated_time = len(tasks) * len(model_configs) * 30  # 30 seconds per task
        print(f"⏱️  Estimated time: {estimated_time//60} minutes {estimated_time%60} seconds")
        
        # 8. Confirm before proceeding
        if not quick_mode:
            confirm = input("\n🤔 Continue with evaluation? (y/N): ")
            if confirm.lower() != 'y':
                print("👋 Evaluation cancelled")
                return
        
        # 9. Run evaluation
        print("\n🚀 Starting real model evaluation...")
        
        def progress_callback(message):
            print(f"   📈 {message}")
        
        try:
            results = await self.evaluation_engine.run_evaluation(
                model_configs,
                eval_config,
                progress_callback
            )
            
            print(f"\n✅ Evaluation completed!")
            print(f"⏱️  Total time: {results.duration:.1f} seconds")
            
            # 10. Display results
            await self.display_results(results)
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def display_results(self, results):
        """Display evaluation results"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)
        
        # Leaderboard
        leaderboard = results.get_leaderboard()
        print("\n🏆 LEADERBOARD:")
        print("-" * 50)
        
        for i, entry in enumerate(leaderboard, 1):
            print(f"{i}. {entry['model_name']}")
            print(f"   Overall Score: {entry['overall_score']:.3f}")
            print(f"   Pass Rate: {entry['pass_rate']:.1%}")
            print(f"   Avg Time: {entry['avg_time']:.1f}s")
            print(f"   Frontend: {entry['frontend_score']:.3f}")
            print(f"   Backend: {entry['backend_score']:.3f}")
            print(f"   Testing: {entry['testing_score']:.3f}")
            print()
        
        # Detailed results for top model
        if results.model_results:
            best_model = results.model_results[0]
            print(f"🔍 DETAILED RESULTS - {best_model.model_name}")
            print("-" * 50)
            
            for task_result in best_model.task_results:
                status = "✅ PASSED" if task_result.passed else "❌ FAILED"
                print(f"{status} {task_result.task_title}")
                print(f"   Score: {task_result.score:.3f}")
                print(f"   Time: {task_result.execution_time:.1f}s")
                print(f"   Code: {len(task_result.generated_code)} chars")
                if task_result.error_message:
                    print(f"   Error: {task_result.error_message}")
                print()
        
        # Save location
        print("💾 RESULTS SAVED:")
        print(f"   JSON: results/results_{results.run_id}.json")
        print(f"   Report: results/report_{results.run_id}.html")
        
        print("\n🎉 Real model evaluation completed!")


def main():
    """Main function with CLI options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run real model evaluation")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (not quick mode)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    evaluator = RealModelEvaluator()
    
    if args.list_models:
        print("🔍 Checking available Ollama models...")
        if not evaluator.check_ollama_connection():
            print("❌ Ollama is not running!")
            return
        
        available = evaluator.get_available_models()
        recommended = evaluator.get_recommended_models()
        
        print(f"\n📦 Available models ({len(available)}):")
        for model in available:
            marker = "🎯" if model in recommended else "  "
            print(f"   {marker} {model}")
        
        print(f"\n🎯 Recommended for coding ({len(recommended)}):")
        for model in recommended:
            print(f"   - {model}")
        
        if not available:
            print("\n📥 Download models with:")
            print("   ollama pull codellama:7b")
            print("   ollama pull deepseek-coder:6.7b")
        
        return
    
    # Run evaluation
    asyncio.run(evaluator.run_real_evaluation(
        selected_models=args.models,
        quick_mode=not args.full
    ))


if __name__ == "__main__":
    main()
