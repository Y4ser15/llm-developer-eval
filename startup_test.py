#!/usr/bin/env python3
"""
Quick Platform Startup Test
Verifies the platform can start without errors.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_platform_startup():
    """Test that the platform can start successfully"""
    logger.info("🚀 Testing platform startup...")
    
    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the main components
        logger.info("📦 Importing core components...")
        from src.core.model_interfaces import ModelConfig, ModelFactory
        logger.info("✅ Model interfaces imported")
        
        from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
        logger.info("✅ Evaluation engine imported")
        
        from src.core.bigcodebench_integration import BenchmarkOrchestrator
        logger.info("✅ Benchmark orchestrator imported")
        
        # Import the web app
        logger.info("🌐 Importing web application...")
        from src.web.comprehensive_app import app
        logger.info("✅ Web application imported")
        
        # Test component initialization
        logger.info("🔧 Testing component initialization...")
        evaluator = ComprehensiveEvaluator()
        logger.info("✅ Evaluator initialized")
        
        orchestrator = BenchmarkOrchestrator()
        logger.info("✅ Orchestrator initialized")
        
        logger.info("🎉 Platform startup test PASSED!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        return False

if __name__ == "__main__":
    success = test_platform_startup()
    if success:
        print("\n🎯 Platform is ready! Start with: python app.py")
    else:
        print("\n❌ Platform needs fixes before starting")
    sys.exit(0 if success else 1)
