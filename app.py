#!/usr/bin/env python3
"""
LLM Coding Evaluation Platform - Main Entry Point
Professional single entry point for the evaluation platform.
"""

import os
import sys
import uvicorn
import argparse
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Set environment variables
os.environ.setdefault("PYTHONPATH", str(project_root))

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Coding Evaluation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Start development server
  python app.py --production       # Start production server  
  python app.py --port 3000        # Custom port
  python app.py --host 0.0.0.0     # Bind to all interfaces
        """
    )
    
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    
    args = parser.parse_args()
    
    # Print startup information
    print("🚀 LLM Coding Evaluation Platform")
    print("=" * 50)
    print(f"🌐 Web Interface: http://{args.host}:{args.port}")
    print(f"📊 Dashboard: http://{args.host}:{args.port}/dashboard") 
    print(f"🔧 Evaluation: http://{args.host}:{args.port}/evaluate")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🏥 Health Check: http://{args.host}:{args.port}/health")
    print("=" * 50)
    print("⏹️  Press Ctrl+C to stop")
    print()
    
    # Configure server
    config = {
        "app": "src.web.simple_app:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload and not args.production,
        "log_level": "info" if args.production else "debug",
        "access_log": args.production,
    }
    
    if args.production:
        config.update({
            "workers": args.workers,
            "reload": False
        })
        print("🏭 Running in PRODUCTION mode")
    else:
        print("🔧 Running in DEVELOPMENT mode")
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n👋 Platform stopped gracefully")
    except Exception as e:
        print(f"\n❌ Error starting platform: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
