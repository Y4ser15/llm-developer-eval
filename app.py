#!/usr/bin/env python3
"""
LLM Coding Evaluation Platform - Main Application
Simple entry point for running the platform locally or in production.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Set environment variables
os.environ.setdefault("PYTHONPATH", str(project_root))

def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Coding Evaluation Platform")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    
    args = parser.parse_args()
    
    print("🚀 Starting LLM Coding Evaluation Platform")
    print(f"🌐 Web interface: http://{args.host}:{args.port}")
    print(f"📊 Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"🔗 API docs: http://{args.host}:{args.port}/docs")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    # Configure uvicorn
    config = {
        "app": "src.web.comprehensive_app:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload and not args.production,
        "log_level": "info" if args.production else "debug",
    }
    
    if args.production:
        # Production settings
        config.update({
            "workers": 1,
            "access_log": True,
            "reload": False
        })
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n👋 Platform stopped")


if __name__ == "__main__":
    main()
