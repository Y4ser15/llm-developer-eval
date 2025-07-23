#!/usr/bin/env python3
"""
Quick test runner to check platform status
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

def run_tests():
    print("🚀 Running LLM Coding Evaluation Platform Tests")
    print("=" * 60)
    
    try:
        # Import and run the main test function
        from test_platform import main
        main()
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
