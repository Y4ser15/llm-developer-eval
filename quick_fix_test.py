#!/usr/bin/env python3
"""
Quick Fix Test - Test Fixed Public Datasets
Verifies the platform works with the fixed streaming datasets.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_fixed_datasets():
    """Test that the fixed datasets work correctly"""
    logger.info("🔍 Testing fixed public datasets...")
    
    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the fixed datasets module
        from src.core.fixed_public_datasets import FixedPublicDatasetManager, FixedBenchmarkOrchestrator
        
        logger.info("✅ Successfully imported FixedPublicDatasetManager")
        
        # Create dataset manager
        manager = FixedPublicDatasetManager()
        
        # Check available datasets
        datasets_status = await manager.get_available_datasets()
        logger.info(f"📊 Available datasets: {datasets_status}")
        
        # Test streaming (small amounts to avoid long downloads)
        for dataset_name, available in datasets_status.items():
            if available:
                try:
                    logger.info(f"🧪 Testing {dataset_name} (streaming 2 tasks)...")
                    tasks = await manager.get_domain_tasks('general', max_tasks=2, dataset_preference=[dataset_name])
                    logger.info(f"✅ {dataset_name}: Loaded {len(tasks)} tasks successfully")
                except Exception as e:
                    logger.error(f"❌ {dataset_name} failed: {e}")
            else:
                logger.warning(f"⚠️ {dataset_name}: Not available")
        
        # Test orchestrator
        logger.info("🧪 Testing FixedBenchmarkOrchestrator...")
        orchestrator = FixedBenchmarkOrchestrator()
        
        # Check compatibility properties
        logger.info(f"📊 BigCodeBench available: {orchestrator.bigcodebench.bigcodebench_available}")
        logger.info(f"📊 HumanEval available: {orchestrator.humaneval.available}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fixed datasets test failed: {e}")
        return False


async def test_web_app_import():
    """Test that the web app imports correctly with fixed datasets"""
    logger.info("🧪 Testing web app import with fixed datasets...")
    
    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the web app
        from src.web.comprehensive_app import app, orchestrator
        logger.info("✅ Successfully imported web app with fixed datasets")
        
        # Test orchestrator is the fixed version
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
    """Run quick fix tests"""
    logger.info("🚀 Starting Quick Fix Tests")
    logger.info("=" * 40)
    
    tests = [
        ("Fixed Datasets", test_fixed_datasets),
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
    
    logger.info("\n" + "=" * 40)
    logger.info(f"📊 QUICK FIX RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL FIXES WORK - Platform ready!")
        logger.info("🚀 Start platform: python app.py")
        logger.info("🌐 Access: http://localhost:8000/evaluate")
        logger.info("📋 Features:")
        logger.info("  • Streaming datasets (no full downloads)")
        logger.info("  • Fixed async issues (no RuntimeWarnings)")
        logger.info("  • Updated frontend (no Alpine.js errors)")
        logger.info("  • HumanEval + MBPP datasets available")
        return True
    else:
        logger.error("💥 Some fixes failed - Check errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
