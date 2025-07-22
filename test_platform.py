#!/usr/bin/env python3
"""
Test script for LLM Coding Evaluation Platform
Run this to verify that all components are working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.core.model_interfaces import ModelFactory, ModelConfig
        from src.core.custom_datasets import DatasetManager, TaskType, DifficultyLevel
        from src.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig
        from src.utils.report_generator import ReportGenerator
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_datasets():
    """Test dataset loading"""
    print("🧪 Testing dataset loading...")
    
    try:
        from src.core.custom_datasets import DatasetManager
        
        manager = DatasetManager()
        summary = manager.export_tasks_summary()
        
        print(f"✅ Loaded {summary['total_tasks']} tasks:")
        print(f"   - Frontend: {summary['by_type']['frontend']} tasks")
        print(f"   - Backend: {summary['by_type']['backend']} tasks") 
        print(f"   - Testing: {summary['by_type']['testing']} tasks")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return False

def test_model_interfaces():
    """Test model interface creation"""
    print("🧪 Testing model interfaces...")
    
    try:
        from src.core.model_interfaces import ModelFactory, ModelConfig
        
        # Test Ollama interface
        config = ModelConfig(
            name="Test Ollama",
            provider="ollama",
            model_name="codellama:7b",
            base_url="http://localhost:11434"
        )
        
        interface = ModelFactory.create_interface(config)
        print("✅ Ollama interface created successfully")
        
        # Test OpenAI interface  
        config = ModelConfig(
            name="Test OpenAI",
            provider="openai", 
            model_name="gpt-4-turbo-preview",
            api_key="test-key"
        )
        
        interface = ModelFactory.create_interface(config)
        print("✅ OpenAI interface created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model interface error: {e}")
        return False

def test_evaluation_engine():
    """Test evaluation engine initialization"""
    print("🧪 Testing evaluation engine...")
    
    try:
        from src.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig
        
        engine = EvaluationEngine()
        config = EvaluationConfig(
            task_types=[TaskType.FRONTEND],
            difficulty_levels=[DifficultyLevel.EASY],
            max_tasks_per_type=1,
            include_bigcodebench=False
        )
        
        print("✅ Evaluation engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Evaluation engine error: {e}")
        return False

def test_web_app():
    """Test web application startup"""
    print("🧪 Testing web application...")
    
    try:
        from src.web.app import app
        print("✅ Web application imported successfully")
        return True
    except Exception as e:
        print(f"❌ Web application error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 LLM Coding Evaluation Platform - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_datasets,
        test_model_interfaces,
        test_evaluation_engine,
        test_web_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Platform is ready to use.")
        print("\n🚀 Quick Start:")
        print("1. Run: python setup.py --mode start")
        print("2. Open: http://localhost:8000")
        print("3. Configure your models and start evaluating!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("💡 Try running: python setup.py --mode setup")

if __name__ == "__main__":
    main()