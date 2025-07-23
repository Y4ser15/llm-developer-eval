#!/usr/bin/env python3
"""
LLM Coding Evaluation Platform - One-Click Deployment Script
Comprehensive setup and deployment automation for all project requirements.
"""

import subprocess
import sys
import os
import argparse
import json
import time
from pathlib import Path
import requests
import yaml


class PlatformDeployer:
    """Comprehensive platform deployment and management"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_met = {}
        
    def check_system_requirements(self):
        """Check all system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 9):
            print(f"✅ Python {python_version.major}.{python_version.minor} (✓)")
            self.requirements_met['python'] = True
        else:
            print(f"❌ Python {python_version.major}.{python_version.minor} (requires 3.9+)")
            self.requirements_met['python'] = False
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker installed (✓)")
                self.requirements_met['docker'] = True
                
                # Check if Docker daemon is running
                result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Docker daemon running (✓)")
                    self.requirements_met['docker_running'] = True
                else:
                    print("⚠️  Docker daemon not running")
                    self.requirements_met['docker_running'] = False
            else:
                print("❌ Docker not installed")
                self.requirements_met['docker'] = False
        except FileNotFoundError:
            print("❌ Docker not found in PATH")
            self.requirements_met['docker'] = False
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker Compose installed (✓)")
                self.requirements_met['docker_compose'] = True
            else:
                print("❌ Docker Compose not installed")
                self.requirements_met['docker_compose'] = False
        except FileNotFoundError:
            print("❌ Docker Compose not found in PATH")
            self.requirements_met['docker_compose'] = False
        
        # Check Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Git installed (✓)")
                self.requirements_met['git'] = True
            else:
                print("❌ Git not installed")
                self.requirements_met['git'] = False
        except FileNotFoundError:
            print("❌ Git not found in PATH")
            self.requirements_met['git'] = False
        
        return all(self.requirements_met.values())
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing Python dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            
            # Install requirements
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
            
            print("✅ Python dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Python dependencies: {e}")
            return False
    
    def setup_bigcodebench(self):
        """Set up BigCodeBench framework"""
        print("\n🔧 Setting up BigCodeBench...")
        
        try:
            # Install BigCodeBench
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'git+https://github.com/bigcode-project/bigcodebench.git'
            ], check=True)
            
            # Test BigCodeBench installation
            result = subprocess.run([
                sys.executable, '-c', 'import bigcodebench; print("BigCodeBench available")'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ BigCodeBench installed and available")
                return True
            else:
                print("❌ BigCodeBench installation failed")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup BigCodeBench: {e}")
            return False
    
    def setup_ollama(self):
        """Set up Ollama for local models"""
        print("\n🤖 Setting up Ollama for local models...")
        
        # Check if Ollama is already running
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is already running")
                return True
        except:
            pass
        
        # Start Ollama via Docker
        try:
            print("🚀 Starting Ollama via Docker...")
            subprocess.run([
                'docker', 'run', '-d', '--name', 'ollama-setup',
                '-p', '11434:11434',
                '-v', 'ollama_data:/root/.ollama',
                'ollama/ollama:latest'
            ], check=True)
            
            # Wait for Ollama to start
            print("⏳ Waiting for Ollama to start...")
            for i in range(30):
                try:
                    response = requests.get('http://localhost:11434/api/tags', timeout=2)
                    if response.status_code == 200:
                        print("✅ Ollama started successfully")
                        break
                except:
                    time.sleep(2)
            else:
                print("⚠️  Ollama may not have started properly")
                return False
            
            # Download recommended models
            models = ['codellama:7b', 'deepseek-coder:6.7b']
            for model in models:
                print(f"📥 Downloading {model}...")
                result = subprocess.run([
                    'docker', 'exec', 'ollama-setup', 'ollama', 'pull', model
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {model} downloaded successfully")
                else:
                    print(f"⚠️  Failed to download {model}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup Ollama: {e}")
            return False
    
    def create_environment_file(self):
        """Create .env file with configuration"""
        print("\n📝 Creating environment configuration...")
        
        env_content = """# LLM Coding Evaluation Platform Environment Configuration

# API Keys (Add your keys here)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Application Configuration
PYTHONPATH=.
LOG_LEVEL=INFO

# Database Configuration (for production)
DATABASE_URL=postgresql://postgres:password@localhost:5432/llm_eval
REDIS_URL=redis://localhost:6379

# BigCodeBench Configuration
BIGCODEBENCH_TIMEOUT=300
BIGCODEBENCH_MAX_WORKERS=4

# Web Server Configuration
HOST=localhost
PORT=8000
DEBUG=false
"""
        
        env_file = self.project_root / '.env'
        if not env_file.exists():
            env_file.write_text(env_content)
            print("✅ Created .env file")
            print("📝 Please edit .env file to add your API keys")
        else:
            print("✅ .env file already exists")
        
        return True
    
    def run_comprehensive_tests(self):
        """Run comprehensive platform tests"""
        print("\n🧪 Running comprehensive platform tests...")
        
        try:
            # Run the comprehensive diagnostic
            result = subprocess.run([
                sys.executable, 'comprehensive_diagnostic.py'
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Tests failed: {e}")
            return False
    
    def deploy_with_docker(self, mode='development'):
        """Deploy platform using Docker"""
        print(f"\n🐳 Deploying platform with Docker ({mode} mode)...")
        
        try:
            if mode == 'production':
                # Build and start with production profile
                subprocess.run([
                    'docker-compose', '--profile', 'production', 'up', '--build', '-d'
                ], check=True)
            else:
                # Build and start development environment
                subprocess.run([
                    'docker-compose', 'up', '--build', '-d'
                ], check=True)
            
            print("✅ Platform deployed successfully with Docker")
            print("🌐 Web interface available at: http://localhost:8000")
            print("📊 Dashboard available at: http://localhost:8000/dashboard")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker deployment failed: {e}")
            return False
    
    def deploy_native(self):
        """Deploy platform natively (without Docker)"""
        print("\n🚀 Deploying platform natively...")
        
        try:
            # Start the web server
            print("Starting web server...")
            subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 
                'src.web.comprehensive_app:app',
                '--host', 'localhost', '--port', '8000', '--reload'
            ])
            
            print("✅ Platform started successfully")
            print("🌐 Web interface available at: http://localhost:8000")
            print("📊 Dashboard available at: http://localhost:8000/dashboard")
            print("⏹️  Press Ctrl+C to stop")
            
            return True
            
        except Exception as e:
            print(f"❌ Native deployment failed: {e}")
            return False
    
    def show_quick_start(self):
        """Show quick start guide"""
        print("\n🎉 DEPLOYMENT COMPLETE!")
        print("=" * 60)
        print("🚀 LLM Coding Evaluation Platform is ready!")
        print()
        print("📖 Quick Start Guide:")
        print("1. Open your browser and go to: http://localhost:8000")
        print("2. Configure your models (Ollama models or API keys)")
        print("3. Select evaluation domains: Frontend, Backend, Testing")
        print("4. Click 'Start Evaluation' for one-click assessment")
        print("5. View results in the dashboard")
        print()
        print("🔧 Configuration:")
        print(f"- Edit .env file to add API keys: {self.project_root / '.env'}")
        print("- Ollama models available at: http://localhost:11434")
        print("- Results saved in: ./results/")
        print()
        print("📚 Documentation:")
        print("- API docs: http://localhost:8000/docs")
        print("- Health check: http://localhost:8000/health")
        print()
        print("🆘 Need help?")
        print("- Check logs: docker-compose logs -f llm-eval-platform")
        print("- Run diagnostics: python comprehensive_diagnostic.py")
        print("- View issues: https://github.com/your-repo/issues")


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="LLM Coding Evaluation Platform Deployment")
    parser.add_argument('--mode', choices=['full', 'docker', 'native', 'test', 'setup'], 
                       default='full', help='Deployment mode')
    parser.add_argument('--skip-ollama', action='store_true', help='Skip Ollama setup')
    parser.add_argument('--skip-tests', action='store_true', help='Skip comprehensive tests')
    parser.add_argument('--production', action='store_true', help='Deploy in production mode')
    
    args = parser.parse_args()
    
    deployer = PlatformDeployer()
    
    print("🚀 LLM Coding Evaluation Platform - One-Click Deployment")
    print("=" * 60)
    print("📋 This script will set up the complete evaluation platform with:")
    print("   • BigCodeBench integration")
    print("   • Multi-LLM support (Ollama, OpenAI, Anthropic)")
    print("   • Web interface with real-time evaluation")
    print("   • Comprehensive benchmarking across 3 domains")
    print("   • Docker deployment with all dependencies")
    print()
    
    success = True
    
    if args.mode in ['full', 'setup']:
        # Check system requirements
        if not deployer.check_system_requirements():
            print("\n❌ System requirements not met. Please install missing dependencies.")
            sys.exit(1)
        
        # Install Python dependencies
        if not deployer.install_python_dependencies():
            success = False
        
        # Setup BigCodeBench
        if not deployer.setup_bigcodebench():
            success = False
        
        # Setup Ollama (optional)
        if not args.skip_ollama:
            if not deployer.setup_ollama():
                print("⚠️  Ollama setup failed, but continuing with other providers")
        
        # Create environment file
        deployer.create_environment_file()
    
    if args.mode in ['full', 'test'] and not args.skip_tests:
        # Run comprehensive tests
        if not deployer.run_comprehensive_tests():
            print("⚠️  Some tests failed, but deployment can continue")
    
    if args.mode in ['full', 'docker']:
        # Deploy with Docker
        mode = 'production' if args.production else 'development'
        if not deployer.deploy_with_docker(mode):
            success = False
    
    elif args.mode == 'native':
        # Deploy natively
        if not deployer.deploy_native():
            success = False
    
    if success and args.mode != 'setup':
        deployer.show_quick_start()
    elif args.mode == 'setup':
        print("\n✅ Setup completed successfully!")
        print("Run with --mode=docker or --mode=native to start the platform")
    else:
        print("\n❌ Deployment completed with errors. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
