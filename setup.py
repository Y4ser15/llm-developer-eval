#!/usr/bin/env python3
"""
LLM Coding Evaluation Platform Setup Script

This script helps you set up the LLM coding evaluation platform quickly.
It can install dependencies, set up Ollama models, and start the web interface.
"""

import subprocess
import sys
import os
import argparse
import json
import time
from pathlib import Path


def run_command(command, check=True, cwd=None):
    """Run a shell command"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True, cwd=cwd
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = run_command("docker --version", check=False)
        if result.returncode == 0:
            print("✅ Docker is installed")

            # Check if Docker daemon is running
            result = run_command("docker info", check=False)
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                return True
            else:
                print("⚠️  Docker daemon is not running. Please start Docker.")
                return False
        else:
            print("⚠️  Docker is not installed")
            return False
    except Exception as e:
        print(f"⚠️  Error checking Docker: {e}")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")

    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Install requirements
    if Path("requirements.txt").exists():
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
        print("✅ Dependencies installed successfully")
    else:
        print("❌ requirements.txt not found")
        sys.exit(1)


def setup_ollama():
    """Set up Ollama and download models"""
    print("\n🚀 Setting up Ollama...")

    # Check if Ollama is installed
    result = run_command("ollama --version", check=False)
    if result.returncode != 0:
        print("❌ Ollama is not installed. Please install it from https://ollama.com/")
        return False

    print("✅ Ollama is installed")

    # Check if Ollama server is running
    result = run_command("curl -s http://localhost:11434/api/tags", check=False)
    if result.returncode != 0:
        print("⚠️  Ollama server is not running. Starting Ollama server...")
        print(
            "Please run 'ollama serve' in another terminal and press Enter to continue..."
        )
        input()

    # Download recommended models
    models = ["codellama:7b", "deepseek-coder:6.7b", "qwen2.5-coder:7b"]

    for model in models:
        print(f"📥 Downloading {model}...")
        result = run_command(f"ollama pull {model}", check=False)
        if result.returncode == 0:
            print(f"✅ {model} downloaded successfully")
        else:
            print(f"⚠️  Failed to download {model}")

    return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")

    directories = ["results", "datasets", "src/web/static", "src/web/templates"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")


def setup_environment():
    """Set up environment variables"""
    print("\n🔧 Setting up environment...")

    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# LLM Coding Evaluation Platform Environment Variables

# API Keys (optional - only needed for cloud models)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Application Configuration
PYTHONPATH=.
LOG_LEVEL=INFO
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")
        print("📝 Please edit .env file to add your API keys if needed")
    else:
        print("✅ .env file already exists")


def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")

    try:
        # Test imports
        sys.path.insert(0, str(Path.cwd()))

        from src.core.model_interfaces import ModelFactory, ModelConfig
        from src.core.custom_datasets import DatasetManager
        from src.evaluation.evaluation_engine import EvaluationEngine

        print("✅ Core modules can be imported")

        # Test dataset loading
        dataset_manager = DatasetManager()
        summary = dataset_manager.export_tasks_summary()
        print(f"✅ Loaded {summary['total_tasks']} tasks from datasets")

        # Test model interface creation
        config = ModelConfig(
            name="Test Model", provider="ollama", model_name="codellama:7b"
        )
        interface = ModelFactory.create_interface(config)
        print("✅ Model interface creation works")

        return True

    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def start_server(mode="development"):
    """Start the web server"""
    print(f"\n🌐 Starting web server in {mode} mode...")

    if mode == "development":
        print("Starting development server...")
        print("🌐 Web interface will be available at: http://localhost:8000")
        print("📖 API documentation will be available at: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server")

        try:
            run_command(
                f"{sys.executable} -m uvicorn src.web.app:app --reload --host localhost --port 8000"
            )
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

    elif mode == "docker":
        if check_docker():
            print("Starting with Docker...")
            run_command("docker-compose up --build")
        else:
            print(
                "❌ Docker is not available. Please install Docker or use development mode."
            )

    elif mode == "production":
        print("Starting production server...")
        run_command(
            f"{sys.executable} -m uvicorn src.web.app:app --host localhost --port 8000"
        )


def main():
    parser = argparse.ArgumentParser(description="LLM Coding Evaluation Platform Setup")
    parser.add_argument(
        "--mode",
        choices=["setup", "start", "docker", "test"],
        default="setup",
        help="Mode to run",
    )
    parser.add_argument(
        "--server-mode",
        choices=["development", "production", "docker"],
        default="development",
        help="Server mode",
    )
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama setup")
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )

    args = parser.parse_args()

    print("🚀 LLM Coding Evaluation Platform Setup")
    print("=" * 50)

    # Check Python version
    check_python_version()

    if args.mode == "setup":
        print("\n📋 Running full setup...")

        # Create directories
        create_directories()

        # Install dependencies
        if not args.skip_deps:
            install_dependencies()

        # Set up environment
        setup_environment()

        # Set up Ollama
        if not args.skip_ollama:
            setup_ollama()

        # Test installation
        if test_installation():
            print("\n🎉 Setup completed successfully!")
            print("\n📖 Quick Start:")
            print("1. Edit .env file to add API keys (optional)")
            print("2. Run 'python setup.py --mode start' to start the server")
            print("3. Open http://localhost:8000 in your browser")
            print("4. Configure your models and start an evaluation!")
        else:
            print("\n❌ Setup completed with errors. Please check the logs above.")

    elif args.mode == "start":
        start_server(args.server_mode)

    elif args.mode == "docker":
        if check_docker():
            print("\n🐳 Starting with Docker...")
            run_command("docker-compose up --build")
        else:
            print("❌ Docker setup failed")

    elif args.mode == "test":
        if test_installation():
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")


if __name__ == "__main__":
    main()
