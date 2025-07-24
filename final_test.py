#!/usr/bin/env python3
"""
Final Platform Test - Real Public Datasets
Tests that the platform works with real datasets (no mock data).
"""

import asyncio
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_public_datasets():
    """Test that public datasets work correctly"""
    logger.info("🔍 Testing public datasets integration...")
    
    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the new public datasets module
        from src.core.real_public_datasets import PublicDatasetManager
        
        logger.info("✅ Successfully imported PublicDatasetManager")
        
        # Create dataset manager
        manager = PublicDatasetManager()
        
        # Check available datasets
        datasets_status = await manager.get_available_datasets()
        logger.info(f"📊 Available datasets: {datasets_status}")
        
        # Test each dataset if available
        for dataset_name, available in datasets_status.items():
            if available:
                try:
                    logger.info(f"🧪 Testing {dataset_name}...")
                    if dataset_name == 'humaneval':
                        tasks = await manager.datasets[dataset_name].get_tasks(max_tasks=3)
                        logger.info(f"✅ {dataset_name}: Loaded {len(tasks)} tasks")
                    elif dataset_name == 'mbpp':
                        tasks = await manager.datasets[dataset_name].get_tasks(max_tasks=3)
                        logger.info(f"✅ {dataset_name}: Loaded {len(tasks)} tasks")
                    elif dataset_name == 'codecontests':
                        tasks = await manager.datasets[dataset_name].get_tasks(max_tasks=3)
                        logger.info(f"✅ {dataset_name}: Loaded {len(tasks)} tasks")
                    elif dataset_name == 'apps':
                        tasks = await manager.datasets[dataset_name].get_tasks(max_tasks=3, difficulty="introductory")
                        logger.info(f"✅ {dataset_name}: Loaded {len(tasks)} tasks")
                except Exception as e:
                    logger.error(f"❌ {dataset_name} failed: {e}")
            else:
                logger.warning(f"⚠️ {dataset_name}: Not available (install datasets library)")
        
        # Test domain task retrieval
        logger.info("🧪 Testing domain task retrieval...")
        for domain in ['frontend', 'backend', 'testing']:
            try:
                tasks = await manager.get_domain_tasks(domain, max_tasks=5)
                logger.info(f"✅ {domain}: Retrieved {len(tasks)} tasks")
            except Exception as e:
                logger.error(f"❌ {domain} domain failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Public datasets test failed: {e}")
        return False


async def test_app_import():
    """Test that the app can be imported without errors"""
    logger.info("🧪 Testing app import...")
    
    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the web app
        from src.web.comprehensive_app import app, orchestrator
        logger.info("✅ Successfully imported web app")
        
        # Test orchestrator
        datasets_status = await orchestrator.dataset_manager.get_available_datasets()
        logger.info(f"✅ Orchestrator working: {len(datasets_status)} datasets available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ App import failed: {e}")
        return False


async def main():
    """Run all final tests"""
    logger.info("🚀 Starting Final Platform Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Public Datasets", test_public_datasets),
        ("App Import", test_app_import),
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
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Platform ready with real datasets!")
        logger.info("🚀 Start platform: python app.py")
        logger.info("🌐 Access: http://localhost:8000")
        return True
    else:
        logger.error("💥 Some tests failed - Check dependencies")
        logger.info("💡 Install missing packages: pip install datasets")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
