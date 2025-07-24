#!/usr/bin/env python3
"""
One-Click Deployment Script for LLM Coding Evaluation Platform
Provides multiple deployment modes with automatic setup.
"""

import os
import sys
import subprocess
import argparse
import time
import requests
from pathlib import Path


def print_banner():
    """Print deployment banner"""
    print("""
🚀 LLM Coding Evaluation Platform
═══════════════════════════════════════════════════
One-click deployment script for comprehensive setup
""")


def check_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "docker": "Docker is required for containerized deployment",
        "docker-compose": "Docker Compose is required for multi-service setup"
    }
    
    missing = []
    for cmd, description in requirements.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            print(f"  ✅ {cmd} - Available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ❌ {cmd} - Missing")
            missing.append(description)
    
    if missing:
        print("\n⚠️  Missing requirements:")
        for req in missing:
            print(f"    - {req}")
        print("\nPlease install missing components and retry.")
        return False
    
    print("✅ All requirements satisfied!")
    return True


def create_env_file():
    """Create .env file with default configuration"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("📝 .env file already exists - keeping existing configuration")
        return
    
    print("📝 Creating .env configuration file...")
    
    env_content = """# LLM Coding Evaluation Platform Configuration

# API Keys (Optional - add your keys here)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_API_KEY=

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Database Configuration
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://postgres:password@postgres:5432/llm_eval

# Application Settings
LOG_LEVEL=INFO
PYTHONPATH=/app

# Production Settings
WORKERS=4
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with default configuration")
    print("💡 Edit .env to add your API keys for cloud models")


def deploy_docker_mode():
    """Deploy using Docker Compose"""
    print("\n🐳 Starting Docker deployment...")
    
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("datasets/frontend", exist_ok=True)
    os.makedirs("datasets/backend", exist_ok=True)
    os.makedirs("datasets/testing", exist_ok=True)
    
    try:
        # Stop any existing services
        print("📦 Stopping existing services...")
        subprocess.run(["docker-compose", "down"], capture_output=True)
        
        # Build and start services
        print("🔨 Building and starting services...")
        result = subprocess.run(
            ["docker-compose", "up", "--build", "-d"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Docker deployment failed:")
            print(result.stderr)
            return False
        
        print("✅ Docker services started successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Docker deployment error: {e}")
        return False


def deploy_production_mode():
    """Deploy in production mode with Nginx"""
    print("\n🏭 Starting production deployment...")
    
    try:
        # Stop existing services
        subprocess.run(["docker-compose", "down"], capture_output=True)
        
        # Start production services
        result = subprocess.run(
            ["docker-compose", "--profile", "production", "up", "--build", "-d"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Production deployment failed:")
            print(result.stderr)
            return False
        
        print("✅ Production services started successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Production deployment error: {e}")
        return False


def deploy_native_mode():
    """Deploy natively with Python"""
    print("\n🐍 Starting native Python deployment...")
    
    try:
        # Check if virtual environment exists
        venv_path = Path("venv")
        if not venv_path.exists():
            print("📦 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
        # Determine activation script
        if sys.platform == "win32":
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_executable = venv_path / "Scripts" / "pip.exe"
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            activate_script = venv_path / "bin" / "activate"
            pip_executable = venv_path / "bin" / "pip"
            python_executable = venv_path / "bin" / "python"
        
        # Install dependencies
        print("📦 Installing dependencies...")
        subprocess.run([str(pip_executable), "install", "-r", "requirements.txt"], check=True)
        
        # Start Redis (required)
        print("🔧 Starting Redis container...")
        subprocess.run([
            "docker", "run", "-d", "--name", "llm-eval-redis", 
            "-p", "6379:6379", "redis:alpine"
        ], capture_output=True)
        
        print("✅ Native setup completed!")
        print(f"🚀 Start the application with: {python_executable} app.py")
        return True
        
    except Exception as e:
        print(f"❌ Native deployment error: {e}")
        return False


def wait_for_service(url, timeout=60, service_name="Platform"):
    """Wait for service to become available"""
    print(f"⏳ Waiting for {service_name} to start...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n⚠️  {service_name} not responding after {timeout}s")
    return False


def setup_ollama_models():
    """Setup recommended Ollama models"""
    print("\n🤖 Setting up Ollama models...")
    
    recommended_models = [
        "codellama:7b",
        "deepseek-coder:6.7b", 
        "qwen2.5-coder:7b"
    ]
    
    # Check if Ollama is available
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Ollama not available - skipping model setup")
            print("💡 Install Ollama manually: https://ollama.ai/")
            return
    except FileNotFoundError:
        print("⚠️  Ollama not installed - skipping model setup")
        return
    
    print("📦 Pulling recommended models (this may take a while)...")
    for model in recommended_models:
        print(f"  📥 Pulling {model}...")
        result = subprocess.run(["ollama", "pull", model], capture_output=True)
        if result.returncode == 0:
            print(f"  ✅ {model} - Ready")
        else:
            print(f"  ⚠️  {model} - Failed to pull")


def show_status():
    """Show deployment status and URLs"""
    print("\n🎉 Deployment completed!")
    print("═" * 50)
    print("🌐 Platform URLs:")
    print("   • Main Interface: http://localhost:8000")
    print("   • Dashboard: http://localhost:8000/dashboard")
    print("   • Evaluation: http://localhost:8000/evaluate")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    
    print("\n🔧 Service Status:")
    services = [
        ("Platform", "http://localhost:8000/health"),
        ("Ollama", "http://localhost:11434/api/tags")
    ]
    
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=5)
            status = "🟢 Running" if response.status_code == 200 else "🟡 Issues"
        except:
            status = "🔴 Not Available"
        print(f"   • {service_name}: {status}")
    
    print("\n📚 Next Steps:")
    print("   1. Open http://localhost:8000 in your browser")
    print("   2. Configure API keys in .env file (for cloud models)")
    print("   3. Go to /evaluate to start your first benchmark")
    print("   4. Monitor results in /dashboard")
    
    print("\n⏹️  To stop: docker-compose down")
    print("═" * 50)


def main():
    """Main deployment orchestrator"""
    parser = argparse.ArgumentParser(
        description="LLM Coding Evaluation Platform Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deployment Modes:
  docker      - Docker Compose deployment (default)
  production  - Production mode with Nginx
  native      - Native Python deployment
  
Examples:
  python deploy.py                    # Default Docker deployment
  python deploy.py --mode production  # Production deployment
  python deploy.py --mode native      # Native Python setup
  python deploy.py --setup-ollama     # Setup Ollama models
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["docker", "production", "native"],
        default="docker",
        help="Deployment mode (default: docker)"
    )
    
    parser.add_argument(
        "--setup-ollama",
        action="store_true",
        help="Setup recommended Ollama models"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true", 
        help="Skip system requirement checks"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check requirements
    if not args.skip_checks and not check_requirements():
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Deploy based on mode
    success = False
    if args.mode == "docker":
        success = deploy_docker_mode()
    elif args.mode == "production":
        success = deploy_production_mode()
    elif args.mode == "native":
        success = deploy_native_mode()
    
    if not success:
        print("\n❌ Deployment failed!")
        sys.exit(1)
    
    # Wait for services
    if args.mode in ["docker", "production"]:
        wait_for_service("http://localhost:8000/health")
    
    # Setup Ollama models if requested
    if args.setup_ollama:
        setup_ollama_models()
    
    # Show final status
    show_status()


if __name__ == "__main__":
    main()
