#!/usr/bin/env python3
"""
Local Development Server for LLM Coding Evaluation Platform
Simple script to run the server locally for development and testing.
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import argparse


def setup_environment():
    """Set up local development environment"""
    print("🔧 Setting up local development environment...")
    
    # Create .env from example if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env from .env.example...")
        env_content = env_example.read_text()
        env_file.write_text(env_content)
        print("✅ Created .env file")
        print("💡 Edit .env file to add your API keys if needed")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example found, creating basic .env...")
        basic_env = """# Basic development environment
PYTHONPATH=.
LOG_LEVEL=INFO
HOST=localhost
PORT=8000
DEBUG=true
OLLAMA_BASE_URL=http://localhost:11434
"""
        env_file.write_text(basic_env)
        print("✅ Created basic .env file")


def install_basic_dependencies():
    """Install basic dependencies for development"""
    print("📦 Installing basic dependencies...")
    
    try:
        # Install core dependencies
        core_deps = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0", 
            "jinja2>=3.1.2",
            "requests>=2.31.0",
            "python-dotenv>=1.0.0",
            "pydantic>=2.4.0"
        ]
        
        for dep in core_deps:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
        
        print("✅ Basic dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_ollama():
    """Check if Ollama is available"""
    print("🤖 Checking Ollama availability...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama running with {len(models)} models")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model['name']}")
            return True
        else:
            print("⚠️  Ollama server not responding properly")
            return False
    except Exception as e:
        print("⚠️  Ollama not available (this is optional)")
        print("💡 To use local models, install Ollama from https://ollama.com")
        return False


def start_development_server(host="localhost", port=8000, reload=True):
    """Start the development server"""
    print(f"🚀 Starting development server on {host}:{port}...")
    print("🌐 Web interface: http://localhost:8000")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("🔗 API docs: http://localhost:8000/docs")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.web.comprehensive_app:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure port 8000 is not in use")
        print("2. Check if all dependencies are installed")
        print("3. Run: pip install -r requirements.txt")


def run_quick_test():
    """Run a quick test to verify setup"""
    print("🧪 Running quick platform test...")
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(Path.cwd()))
        
        # Test basic imports
        from src.core.model_interfaces import ModelFactory, ModelConfig
        from src.core.custom_datasets import DatasetManager
        print("✅ Core modules imported successfully")
        
        # Test dataset loading
        manager = DatasetManager()
        summary = manager.export_tasks_summary()
        print(f"✅ Loaded {summary['total_tasks']} tasks from datasets")
        
        # Test model interface creation (with dummy config)
        config = ModelConfig(
            name="Test Model",
            provider="ollama", 
            model_name="test-model"
        )
        interface = ModelFactory.create_interface(config)
        print("✅ Model interface creation works")
        
        print("🎉 Quick test passed! Platform is ready to run.")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False


def main():
    parser = argparse.ArgumentParser(description="Local development server for LLM Evaluation Platform")
    parser.add_argument("--setup", action="store_true", help="Set up development environment")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--install-deps", action="store_true", help="Install basic dependencies")
    
    args = parser.parse_args()
    
    print("🚀 LLM Coding Evaluation Platform - Local Development")
    print("=" * 60)
    
    if args.setup:
        setup_environment()
        return
    
    if args.install_deps:
        if not install_basic_dependencies():
            sys.exit(1)
        return
    
    if args.test:
        if not run_quick_test():
            sys.exit(1)
        return
    
    # Default: start the server
    setup_environment()
    check_ollama()
    
    # Quick test before starting server
    if not run_quick_test():
        print("\n⚠️  Quick test failed, but starting server anyway...")
        print("💡 Some features may not work properly")
    
    start_development_server(
        host=args.host, 
        port=args.port, 
        reload=not args.no_reload
    )


if __name__ == "__main__":
    main()
