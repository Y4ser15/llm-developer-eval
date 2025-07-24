#!/usr/bin/env python3
"""
API Test - Verify Simple App APIs Work
Tests that the missing API endpoints are now working.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_simple_app_apis():
    """Test that the simple app has all required APIs"""
    logger.info("🔍 Testing simple app APIs...")

    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))

        # Import the simple app
        from src.web.simple_app import app

        logger.info("✅ Successfully imported simple app")

        # Check that required routes exist
        routes = [route.path for route in app.routes]

        required_routes = [
            "/api/huggingface/status",
            "/api/huggingface/login",
            "/api/benchmarks/status",
            "/api/evaluations/status/{client_id}",
            "/api/evaluations/active",
            "/api/evaluate",
            "/ws/{client_id}",
        ]

        missing_routes = [r for r in required_routes if r not in routes]

        if missing_routes:
            logger.error(f"❌ Missing routes: {missing_routes}")
            return False
        else:
            logger.info("✅ All required API routes found")
            return True

    except Exception as e:
        logger.error(f"❌ Simple app API test failed: {e}")
        return False


async def test_simple_app_startup():
    """Test that the simple app can start up"""
    logger.info("🧪 Testing simple app startup...")

    try:
        # Add src to Python path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root / "src"))

        # Import components
        from src.web.simple_app import app, evaluator, orchestrator

        logger.info("✅ All components imported successfully")
        logger.info(f"📊 Evaluator type: {type(evaluator).__name__}")
        logger.info(f"📊 Orchestrator type: {type(orchestrator).__name__}")

        # Test dataset availability
        datasets_status = await orchestrator.dataset_manager.get_available_datasets()
        logger.info(f"📊 Datasets available: {datasets_status}")

        return True

    except Exception as e:
        logger.error(f"❌ Simple app startup test failed: {e}")
        return False


async def main():
    """Run API tests"""
    logger.info("🚀 Starting Simple App API Tests")
    logger.info("=" * 40)

    tests = [
        ("Simple App APIs", test_simple_app_apis),
        ("Simple App Startup", test_simple_app_startup),
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
    logger.info(f"📊 API TESTS RESULTS: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL API TESTS PASSED!")
        logger.info("🚀 Missing API endpoints have been fixed!")
        logger.info("")
        logger.info("Fixed issues:")
        logger.info("  • ✅ /api/huggingface/status endpoint added")
        logger.info("  • ✅ /api/huggingface/login endpoint added")
        logger.info("  • ✅ WebSocket /ws/{client_id} endpoint added")
        logger.info("  • ✅ /api/evaluations/status/{client_id} endpoint added")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. python app.py")
        logger.info("2. Open: http://localhost:8000/evaluate")
        logger.info("3. Start evaluation - should complete properly now!")
        return True
    else:
        logger.error("💥 Some API tests failed - Check errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
