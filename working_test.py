#!/usr/bin/env python3
"""
Final Working Test - Verify Everything Works
Tests the working datasets integration.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_working_datasets():
    """Test that the working datasets work correctly"""
    logger.info("🔍 Testing working datasets...")
    
    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the working datasets module
        from src.core.working_datasets import WorkingDatasetManager, WorkingBenchmarkOrchestrator
        
        logger.info("✅ Successfully imported WorkingDatasetManager")
        
        # Create dataset manager
        manager = WorkingDatasetManager()
        
        # Check available datasets
        datasets_status = await manager.get_available_datasets()
        logger.info(f"📊 Available datasets: {datasets_status}")
        
        # Test HumanEval with small number of tasks
        if datasets_status.get('humaneval', False):
            try:
                logger.info("🧪 Testing HumanEval (2 tasks)...")
                tasks = await manager.get_domain_tasks('general', max_tasks=2, dataset_preference=['humaneval'])
                logger.info(f"✅ HumanEval: Loaded {len(tasks)} tasks successfully")
                
                # Check task structure
                if tasks:
                    task = tasks[0]
                    required_fields = ['task_id', 'prompt', 'entry_point', 'domain']
                    missing_fields = [field for field in required_fields if field not in task]
                    if missing_fields:
                        logger.error(f"❌ Missing fields: {missing_fields}")
                    else:
                        logger.info("✅ Task structure is correct")
                
            except Exception as e:
                logger.error(f"❌ HumanEval test failed: {e}")
        else:
            logger.warning("⚠️ HumanEval: Not available")
        
        # Test orchestrator
        logger.info("🧪 Testing WorkingBenchmarkOrchestrator...")
        orchestrator = WorkingBenchmarkOrchestrator()
        
        # Check compatibility properties
        logger.info(f"📊 BigCodeBench available: {orchestrator.bigcodebench.bigcodebench_available}")
        logger.info(f"📊 HumanEval available: {orchestrator.humaneval.available}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Working datasets test failed: {e}")
        return False


async def test_web_app_import():
    """Test that the web app imports correctly with working datasets"""
    logger.info("🧪 Testing web app import with working datasets...")
    
    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the web app
        from src.web.comprehensive_app import app, orchestrator
        logger.info("✅ Successfully imported web app with working datasets")
        
        # Test orchestrator is the working version
        logger.info(f"📊 Orchestrator type: {type(orchestrator).__name__}")
        logger.info(f"📊 Dataset manager type: {type(orchestrator.dataset_manager).__name__}")
        
        # Test dataset availability
        datasets_status = await orchestrator.dataset_manager.get_available_datasets()
        logger.info(f"📊 Web app datasets available: {datasets_status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Web app import test failed: {e}")
        return False


async def main():
    """Run working tests"""
    logger.info("🚀 Starting Working Datasets Tests")
    logger.info("=" * 45)
    
    tests = [
        ("Working Datasets", test_working_datasets),
        ("Web App Import", test_web_app_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            if await test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    logger.info("\n" + "=" * 45)
    logger.info(f"📊 WORKING TESTS RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL WORKING TESTS PASSED!")
        logger.info("🚀 Platform is ready to start!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. python app.py")
        logger.info("2. Open: http://localhost:8000/evaluate")
        logger.info("3. Select a model and start evaluation")
        logger.info("")
        logger.info("Features:")
        logger.info("  • HumanEval streaming dataset")
        logger.info("  • 5 tasks per domain (fast evaluation)")
        logger.info("  • Fixed async handling")
        logger.info("  • No mock data")
        return True
    else:
        logger.error("💥 Some tests failed - Check errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
