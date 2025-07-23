#!/usr/bin/env python3
"""
Comprehensive diagnostic script for LLM Coding Evaluation Platform
This script will test all components and identify what's working vs what needs fixing.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

class PlatformDiagnostic:
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
        self.total_tests += 1
        print(f"\n🔍 Testing {test_name}...")
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                self.passed_tests += 1
                self.results[test_name] = "PASSED"
                return True
            else:
                print(f"❌ {test_name} FAILED")
                self.results[test_name] = "FAILED"
                return False
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            print(f"   Details: {traceback.format_exc()}")
            self.results[test_name] = f"ERROR: {e}"
            return False
    
    def test_basic_imports(self):
        """Test that all basic modules can be imported"""
        try:
            from src.core.model_interfaces import ModelFactory, ModelConfig
            from src.core.custom_datasets import DatasetManager, TaskType, DifficultyLevel
            from src.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig
            from src.utils.report_generator import ReportGenerator
            return True
        except ImportError as e:
            print(f"   Import error: {e}")
            return False
    
    def test_dataset_loading(self):
        """Test dataset loading and task retrieval"""
        try:
            from src.core.custom_datasets import DatasetManager, TaskType
            
            manager = DatasetManager()
            summary = manager.export_tasks_summary()
            
            print(f"   📊 Dataset Summary:")
            print(f"      Total tasks: {summary['total_tasks']}")
            print(f"      Frontend: {summary['by_type']['frontend']}")
            print(f"      Backend: {summary['by_type']['backend']}")
            print(f"      Testing: {summary['by_type']['testing']}")
            
            # Test getting tasks by type
            frontend_tasks = manager.get_tasks_by_type(TaskType.FRONTEND)
            print(f"   📋 Frontend tasks loaded: {len(frontend_tasks)}")
            
            if len(frontend_tasks) > 0:
                task = frontend_tasks[0]
                print(f"   📝 Sample task: {task.title} ({task.difficulty})")
            
            return summary['total_tasks'] >= 9  # Should have 9 tasks minimum
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def test_model_interfaces(self):
        """Test model interface creation"""
        try:
            from src.core.model_interfaces import ModelFactory, ModelConfig
            
            # Test Ollama interface
            ollama_config = ModelConfig(
                name="Test Ollama",
                provider="ollama",
                model_name="codellama:7b",
                base_url="http://localhost:11434"
            )
            ollama_interface = ModelFactory.create_interface(ollama_config)
            print(f"   ✅ Ollama interface created: {ollama_interface.name}")
            
            # Test OpenAI interface
            openai_config = ModelConfig(
                name="Test OpenAI",
                provider="openai",
                model_name="gpt-4-turbo-preview",
                api_key="test-key"
            )
            openai_interface = ModelFactory.create_interface(openai_config)
            print(f"   ✅ OpenAI interface created: {openai_interface.name}")
            
            # Test Anthropic interface
            anthropic_config = ModelConfig(
                name="Test Anthropic",
                provider="anthropic",
                model_name="claude-3-sonnet-20240229",
                api_key="test-key"
            )
            anthropic_interface = ModelFactory.create_interface(anthropic_config)
            print(f"   ✅ Anthropic interface created: {anthropic_interface.name}")
            
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def test_evaluation_engine_init(self):
        """Test evaluation engine initialization"""
        try:
            from src.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig
            from src.core.custom_datasets import TaskType, DifficultyLevel
            
            # Test engine creation
            engine = EvaluationEngine()
            print(f"   ✅ Evaluation engine created")
            
            # Test configuration
            config = EvaluationConfig(
                task_types=[TaskType.FRONTEND, TaskType.BACKEND],
                difficulty_levels=[DifficultyLevel.EASY],
                max_tasks_per_type=1,
                include_bigcodebench=False,
                parallel_execution=False
            )
            print(f"   ✅ Evaluation config created")
            print(f"      Task types: {[t.value for t in config.task_types]}")
            print(f"      Difficulty levels: {[d.value for d in config.difficulty_levels]}")
            
            # Test task selection
            tasks = engine._select_tasks(config)
            print(f"   ✅ Task selection works: {len(tasks)} tasks selected")
            
            for task in tasks:
                print(f"      - {task.title} ({task.task_type.value}, {task.difficulty.value})")
            
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def test_web_app_import(self):
        """Test web application import"""
        try:
            from src.web.app import app
            print(f"   ✅ FastAPI app imported successfully")
            print(f"   📱 App title: {app.title}")
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def test_bigcodebench_integration(self):
        """Test BigCodeBench integration"""
        try:
            from src.core.bigcodebench_integration import CustomBigCodeBenchRunner, BigCodeBenchConfig
            
            runner = CustomBigCodeBenchRunner()
            print(f"   ✅ BigCodeBench runner created")
            print(f"   📦 BigCodeBench available: {runner.bigcodebench_available}")
            
            config = BigCodeBenchConfig()
            print(f"   ✅ BigCodeBench config created")
            
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def test_end_to_end_simulation(self):
        """Test a complete evaluation simulation (without actual model calls)"""
        try:
            from src.core.model_interfaces import ModelConfig, ModelFactory
            from src.core.custom_datasets import DatasetManager, TaskType, DifficultyLevel
            from src.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig
            
            # Create a mock evaluation setup
            engine = EvaluationEngine()
            dataset_manager = DatasetManager()
            
            # Get one easy frontend task
            frontend_tasks = dataset_manager.get_tasks_by_type(TaskType.FRONTEND)
            easy_tasks = [t for t in frontend_tasks if t.difficulty == DifficultyLevel.EASY]
            
            if not easy_tasks:
                print("   ❌ No easy frontend tasks found")
                return False
            
            test_task = easy_tasks[0]
            print(f"   📝 Test task selected: {test_task.title}")
            print(f"   📋 Task prompt length: {len(test_task.prompt)} characters")
            print(f"   🔧 Task has {len(test_task.test_cases)} test cases")
            
            # Create model config (not actually used for generation)
            model_config = ModelConfig(
                name="Mock Model",
                provider="ollama", 
                model_name="test-model"
            )
            
            print(f"   ✅ End-to-end components initialized successfully")
            return True
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def test_file_structure(self):
        """Test that all required files and directories exist"""
        try:
            required_files = [
                "src/core/model_interfaces.py",
                "src/core/custom_datasets.py", 
                "src/core/bigcodebench_integration.py",
                "src/evaluation/evaluation_engine.py",
                "src/utils/report_generator.py",
                "src/web/app.py",
                "requirements.txt",
                "setup.py"
            ]
            
            required_dirs = [
                "src/core",
                "src/evaluation", 
                "src/utils",
                "src/web",
                "src/web/templates"
            ]
            
            missing_files = []
            missing_dirs = []
            
            for file_path in required_files:
                if not (project_root / file_path).exists():
                    missing_files.append(file_path)
            
            for dir_path in required_dirs:
                if not (project_root / dir_path).exists():
                    missing_dirs.append(dir_path)
            
            if missing_files:
                print(f"   ❌ Missing files: {missing_files}")
            if missing_dirs:
                print(f"   ❌ Missing directories: {missing_dirs}")
            
            if not missing_files and not missing_dirs:
                print(f"   ✅ All required files and directories exist")
                return True
            
            return False
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("🚀 LLM Coding Evaluation Platform - Comprehensive Diagnostics")
        print("=" * 70)
        
        # Define all tests
        tests = [
            ("File Structure", self.test_file_structure),
            ("Basic Imports", self.test_basic_imports),
            ("Dataset Loading", self.test_dataset_loading),
            ("Model Interfaces", self.test_model_interfaces),
            ("Evaluation Engine Init", self.test_evaluation_engine_init),
            ("Web App Import", self.test_web_app_import),
            ("BigCodeBench Integration", self.test_bigcodebench_integration),
            ("End-to-End Simulation", self.test_end_to_end_simulation),
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("\n" + "=" * 70)
        print(f"📊 DIAGNOSTIC SUMMARY")
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            print(f"   {status_icon} {test_name}: {result}")
        
        # Recommendations
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED!")
            print("Your platform is ready for evaluation!")
            print("\n🚀 Next Steps:")
            print("1. Start web server: python setup.py --mode start")
            print("2. Configure models and run evaluations")
            print("3. View results at http://localhost:8000")
        else:
            print("\n⚠️  Some tests failed. Recommendations:")
            
            if "FAILED" in str(self.results.get("Basic Imports", "")):
                print("- Fix import issues in core modules")
            
            if "FAILED" in str(self.results.get("Dataset Loading", "")):
                print("- Check dataset initialization")
            
            if "FAILED" in str(self.results.get("Evaluation Engine Init", "")):
                print("- Fix evaluation engine configuration")
            
            print("\n💡 After fixing issues, re-run this diagnostic")


def main():
    """Main diagnostic function"""
    diagnostic = PlatformDiagnostic()
    diagnostic.run_all_tests()


if __name__ == "__main__":
    main()
