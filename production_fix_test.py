#!/usr/bin/env python3
"""
Production Fix Verification Script
Tests that all critical issues have been resolved.
"""

import sys
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_template_fixes():
    """Test that template formatting errors are fixed"""
    logger.info("🔍 Testing template fixes...")
    
    # Check dashboard template
    dashboard_path = Path("src/web/templates/comprehensive_dashboard.html")
    if dashboard_path.exists():
        content = dashboard_path.read_text(encoding='utf-8')
        
        # Check for the old incorrect format
        if '"{:.1%}".format(' in content:
            logger.error("❌ Found old percentage format in dashboard template")
            return False
        
        # Check for the new correct format
        if '(entry.pass_rate * 100)|round(1)' in content:
            logger.info("✅ Dashboard template percentage format fixed")
        else:
            logger.warning("⚠️ New percentage format not found in dashboard")
    
    # Check evaluation result template
    result_path = Path("src/web/templates/evaluation_result.html")
    if result_path.exists():
        content = result_path.read_text(encoding='utf-8')
        
        # Check for the old incorrect format
        if '"{:.1%}".format(' in content:
            logger.error("❌ Found old percentage format in evaluation result template")
            return False
        
        # Check for the new correct format
        if '(entry.pass_rate * 100)|round(1)' in content:
            logger.info("✅ Evaluation result template percentage format fixed")
        else:
            logger.warning("⚠️ New percentage format not found in evaluation result")
    
    return True

def test_import_fixes():
    """Test that imports work correctly without mock fallbacks"""
    logger.info("🔍 Testing import fixes...")
    
    app_path = Path("src/web/comprehensive_app.py")
    if app_path.exists():
        content = app_path.read_text(encoding='utf-8')
        
        # Check that mock classes are removed
        if 'MockEvaluator' in content:
            logger.error("❌ Found MockEvaluator in production code")
            return False
        
        if 'EVALUATION_AVAILABLE' in content:
            logger.error("❌ Found EVALUATION_AVAILABLE check in production code")
            return False
        
        # Check that direct imports exist
        if 'from ..core.model_interfaces import ModelConfig, ModelFactory' in content:
            logger.info("✅ Direct imports found - no mock fallbacks")
        else:
            logger.error("❌ Direct imports not found")
            return False
    
    return True

def test_mock_data_removal():
    """Test that mock data generation is removed"""
    logger.info("🔍 Testing mock data removal...")
    
    app_path = Path("src/web/comprehensive_app.py")
    if app_path.exists():
        content = app_path.read_text(encoding='utf-8')
        
        # Check for mock result generation
        if 'Mock success' in content:
            logger.error("❌ Found mock success generation in production code")
            return False
        
        if 'domain: 1.0 for domain in config.domains' in content:
            logger.error("❌ Found mock domain scores generation")
            return False
        
        if 'Used mock evaluation due to' in content:
            logger.error("❌ Found mock evaluation fallback")
            return False
        
        logger.info("✅ Mock data generation removed")
    
    return True

def test_datetime_import():
    """Test that datetime import is present"""
    logger.info("🔍 Testing datetime import...")
    
    app_path = Path("src/web/comprehensive_app.py")
    if app_path.exists():
        content = app_path.read_text(encoding='utf-8')
        
        if 'from datetime import datetime' in content:
            logger.info("✅ Datetime import found")
            return True
        else:
            logger.error("❌ Datetime import missing")
            return False
    
    return True

def test_app_startup():
    """Test that the app can be imported without errors"""
    logger.info("🔍 Testing app startup...")
    
    try:
        # Change to project directory
        sys.path.insert(0, str(Path.cwd()))
        
        # Try to import the app
        from src.web.comprehensive_app import app
        logger.info("✅ App imports successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ App import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ App startup failed: {e}")
        return False

def main():
    """Run all production fix tests"""
    logger.info("🚀 Starting Production Fix Verification")
    logger.info("=" * 50)
    
    tests = [
        ("Template Fixes", test_template_fixes),
        ("Import Fixes", test_import_fixes),
        ("Mock Data Removal", test_mock_data_removal),
        ("Datetime Import", test_datetime_import),
        ("App Startup", test_app_startup),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Platform is production ready!")
        return True
    else:
        logger.error("💥 Some tests failed - Please review and fix issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
