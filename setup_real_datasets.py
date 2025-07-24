#!/usr/bin/env python3
"""
Setup Real Public Datasets - PRODUCTION READY
Ensures all dependencies are installed for real dataset evaluation.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_package(package):
    """Install a Python package"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      check=True, capture_output=True, text=True)
        logger.info(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package}: {e}")
        return False


def check_and_install_dependencies():
    """Check and install required dependencies for public datasets"""
    
    logger.info("🔍 Checking dependencies for public datasets...")
    
    required_packages = [
        "datasets",  # For HumanEval, MBPP, CodeContests, APPS
        "transformers",  # Often required by datasets
        "torch",  # Often required by datasets
    ]
    
    installed = 0
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} already installed")
            installed += 1
        except ImportError:
            logger.info(f"📦 Installing {package}...")
            if install_package(package):
                installed += 1
    
    logger.info(f"📊 Dependencies: {installed}/{len(required_packages)} installed")
    return installed == len(required_packages)


def test_dataset_access():
    """Test access to public datasets"""
    
    logger.info("🧪 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Test datasets
        test_datasets = [
            ("HumanEval", "openai_humaneval"),
            ("MBPP", "mbpp"),
            ("CodeContests", "deepmind/code_contests"),
            ("APPS", "codeparrot/apps")
        ]
        
        working = 0
        
        for name, dataset_id in test_datasets:
            try:
                logger.info(f"🔍 Testing {name}...")
                # Just check if we can access the dataset info
                dataset = load_dataset(dataset_id, split="test", streaming=True)
                next(iter(dataset))  # Try to get first item
                logger.info(f"✅ {name}: Accessible")
                working += 1
            except Exception as e:
                logger.warning(f"⚠️ {name}: {str(e)[:100]}...")
        
        logger.info(f"📊 Dataset Access: {working}/{len(test_datasets)} working")
        return working > 0  # At least one dataset should work
        
    except ImportError:
        logger.error("❌ datasets library not available")
        return False


def main():
    """Setup public datasets for production use"""
    
    logger.info("🚀 Setting up Real Public Datasets for LLM Evaluation")
    logger.info("=" * 60)
    
    # Step 1: Install dependencies
    logger.info("📦 Step 1: Installing dependencies...")
    if not check_and_install_dependencies():
        logger.error("❌ Failed to install required dependencies")
        return False
    
    # Step 2: Test dataset access
    logger.info("\n🧪 Step 2: Testing dataset access...")
    if not test_dataset_access():
        logger.error("❌ No datasets accessible")
        logger.info("💡 This may be due to network issues or dataset availability")
        logger.info("💡 The platform will still work, but with limited datasets")
    
    # Step 3: Final status
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Setup Complete!")
    logger.info("📋 Available Datasets:")
    logger.info("   • HumanEval: 164 Python programming problems")
    logger.info("   • MBPP: 974 basic Python programming problems")
    logger.info("   • CodeContests: Programming contest problems") 
    logger.info("   • APPS: Python programming with test cases")
    logger.info("")
    logger.info("🚀 Next Steps:")
    logger.info("   1. python app.py")
    logger.info("   2. Open: http://localhost:8000")
    logger.info("   3. Go to /evaluate page")
    logger.info("   4. Select models and start evaluation")
    logger.info("")
    logger.info("✨ Features:")
    logger.info("   • Real datasets (no mock data)")
    logger.info("   • No authentication required")
    logger.info("   • Domain-specific evaluation")
    logger.info("   • Real-time progress tracking")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
