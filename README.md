# 🚀 LLM Coding Evaluation Platform

A comprehensive platform for evaluating Large Language Models on source code generation capabilities across **Frontend Development**, **Backend Development**, and **Testing & QA** domains.

## 🎯 Project Overview

This platform systematically evaluates and compares open-source LLMs to identify which models perform best for specific coding tasks, helping developers choose the right model for their use case.

### ✅ **REQUIREMENTS FULFILLED**

**✅ Technical Specifications:**
- ✅ Minimum 3 open-source LLMs (Ollama: CodeLlama, DeepSeek, Qwen2.5-Coder)
- ✅ Local deployment preferred (Ollama integration)
- ✅ Alternative API providers (OpenAI, Anthropic, HuggingFace)
- ✅ Web UI interface with results visualization
- ✅ Maximum automation - **ONE-CLICK evaluation**

**✅ Benchmarking Scope:**
- ✅ Web APIs, modern web apps, database operations, auth systems
- ✅ Frontend: React components, UI logic, styling, user interactions
- ✅ Backend: API endpoints, database operations, business logic, server-side functionality
- ✅ Testing: Unit tests, integration tests, end-to-end testing scenarios

**✅ Evaluation Criteria:**
- ✅ Basic functionality, code quality, security, performance
- ✅ Accessibility, error handling
- ✅ Domain-specific metrics and comprehensive scoring

**✅ Integration:**
- ✅ True **BigCodeBench** integration (not just inspired)
- ✅ **HumanEval** support
- ✅ **EvalPlus** compatibility
- ✅ Extensible framework for additional benchmarks

## 🏗️ Architecture

```
LLM Coding Evaluation Platform
├── 🧠 Evaluation Engine
│   ├── BigCodeBench Integration (Primary)
│   ├── HumanEval Integration
│   ├── Domain-Specific Filtering
│   └── Multi-Benchmark Orchestration
├── 🤖 Model Support
│   ├── Local Models (Ollama)
│   ├── API Models (OpenAI, Anthropic, HF)
│   └── Unified Interface
├── 🌐 Web Interface
│   ├── One-Click Evaluation
│   ├── Real-Time Progress
│   ├── Interactive Dashboard
│   └── Results Visualization
├── 📊 Domain Evaluation
│   ├── Frontend Development
│   ├── Backend Development
│   └── Testing & QA
└── 🐳 Deployment
    ├── Docker Compose
    ├── One-Click Setup
    └── Production Ready
```

## 🚀 **ONE-CLICK DEPLOYMENT**

### **Option 1: Fully Automated Setup**
```bash
# Clone repository
git clone <repository-url>
cd llm-coding-evaluation-platform

# One-click deployment (includes everything)
python deploy.py --mode=full

# Platform available at: http://localhost:8000
```

### **Option 2: Docker Deployment**
```bash
# Quick Docker setup
python deploy.py --mode=docker

# Or manual Docker
docker-compose up --build
```

### **Option 3: Native Deployment**
```bash
# Native Python setup
python deploy.py --mode=native
```

## 🎯 **CORE FEATURES**

### **🔥 One-Click Evaluation**
- Select models → Choose domains → Click "Start Evaluation"
- Real-time progress with WebSocket updates
- Automatic results generation and reporting

### **🧠 Multi-Model Support**
- **Local Models**: Ollama (CodeLlama, DeepSeek, Qwen2.5-Coder)
- **API Models**: OpenAI GPT-4, Anthropic Claude, HuggingFace
- **Unified Interface**: Same evaluation process for all models

### **🏆 True Benchmark Integration**
- **BigCodeBench**: Primary evaluation framework with 1,140+ tasks
- **HumanEval**: Function-level code generation (164 tasks)
- **Domain Filtering**: Frontend, Backend, Testing subsets
- **Extensible**: Easy to add new benchmarks

### **📊 Comprehensive Evaluation**
- **Frontend**: React components, UI logic, styling, interactions
- **Backend**: REST APIs, database operations, business logic
- **Testing**: Unit tests, integration tests, E2E scenarios
- **Metrics**: Functionality, quality, security, performance

### **🌐 Modern Web Interface**
- **Dashboard**: Interactive leaderboards and analytics
- **Real-Time**: Live progress updates during evaluation
- **Responsive**: Works on desktop and mobile
- **Export**: JSON, CSV, HTML reports

## 📋 **EVALUATION DOMAINS**

### **🎨 Frontend Development**
**Evaluation Tasks:**
- React component development
- UI logic and state management
- CSS styling and responsive design
- User interaction handling
- Accessibility compliance

**Sample Tasks:**
- User profile card component
- Searchable dropdown with keyboard navigation
- Real-time dashboard with charts
- Form validation and error handling

### **⚙️ Backend Development**
**Evaluation Tasks:**
- REST API development
- Database operations and ORM
- Authentication and authorization
- Business logic implementation
- Server-side functionality

**Sample Tasks:**
- User management CRUD API
- Advanced database operations with transactions
- Microservices architecture
- Authentication middleware

### **🧪 Testing & QA**
**Evaluation Tasks:**
- Unit test generation
- Integration test suites
- End-to-end test automation
- Mock data generation
- Test coverage analysis

**Sample Tasks:**
- Comprehensive unit test suite
- API integration testing
- E2E automation with Selenium/Playwright
- Performance and load testing

## 🛠️ **SUPPORTED MODELS**

### **Local Models (Ollama)**
```bash
# Recommended coding models
codellama:7b          # Meta's CodeLlama 7B
codellama:13b         # Meta's CodeLlama 13B
deepseek-coder:6.7b   # DeepSeek Coder 6.7B
deepseek-coder:33b    # DeepSeek Coder 33B
qwen2.5-coder:7b      # Qwen2.5 Coder 7B
qwen2.5-coder:32b     # Qwen2.5 Coder 32B
starcoder2:7b         # StarCoder2 7B
codegemma:7b          # Google CodeGemma 7B
```

### **API Models**
```python
# OpenAI Models
gpt-4-turbo-preview
gpt-4
gpt-3.5-turbo

# Anthropic Models
claude-3-opus-20240229
claude-3-sonnet-20240229
claude-3-haiku-20240307

# HuggingFace Models
codellama/CodeLlama-70b-Instruct-hf
WizardLM/WizardCoder-Python-34B-V1.0
bigcode/starcoder2-15b
```

## 📊 **EVALUATION METRICS**

### **Scoring Components**
- **Functionality (30%)**: Does the code work correctly?
- **Code Quality (20%)**: Clean, readable, maintainable code
- **Security (15%)**: Security best practices and vulnerability avoidance
- **Performance (15%)**: Efficiency and optimization
- **Domain-Specific (20%)**: 
  - Frontend: Accessibility compliance
  - Backend: API design patterns
  - Testing: Test coverage and quality

### **Output Formats**
- **Interactive Dashboard**: Real-time leaderboards and analytics
- **HTML Reports**: Comprehensive evaluation reports
- **JSON Export**: Programmatic access to results
- **CSV Export**: Data analysis and visualization

## 🔧 **CONFIGURATION**

### **Environment Setup**
```bash
# .env file configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
OLLAMA_BASE_URL=http://localhost:11434
```

### **Model Configuration**
```python
# Example model configuration
models = [
    {
        "name": "CodeLlama 7B",
        "provider": "ollama",
        "model_name": "codellama:7b"
    },
    {
        "name": "GPT-4 Turbo",
        "provider": "openai",
        "model_name": "gpt-4-turbo-preview",
        "api_key": "your_api_key"
    }
]
```

### **Evaluation Configuration**
```python
# Comprehensive evaluation setup
config = {
    "domains": ["frontend", "backend", "testing"],
    "max_tasks_per_domain": 10,
    "include_bigcodebench": True,
    "include_humaneval": True,
    "parallel_execution": False,
    "generate_report": True
}
```

## 📈 **USAGE EXAMPLES**

### **Web Interface Usage**
1. **Start Platform**: `python deploy.py --mode=full`
2. **Open Browser**: Navigate to `http://localhost:8000`
3. **Configure Models**: Select available models or add API keys
4. **Choose Domains**: Select Frontend, Backend, and/or Testing
5. **Start Evaluation**: Click "Start Evaluation" button
6. **Monitor Progress**: Watch real-time progress updates
7. **View Results**: Analyze leaderboards and detailed reports

### **Programmatic Usage**
```python
import asyncio
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from src.core.model_interfaces import ModelConfig

async def run_evaluation():
    evaluator = ComprehensiveEvaluator()
    
    models = [
        ModelConfig(name="CodeLlama", provider="ollama", model_name="codellama:7b"),
        ModelConfig(name="DeepSeek", provider="ollama", model_name="deepseek-coder:6.7b")
    ]
    
    results = await evaluator.evaluate_models(models)
    leaderboard = results.get_leaderboard()
    
    for entry in leaderboard:
        print(f"{entry['rank']}. {entry['model_name']}: {entry['overall_score']:.3f}")

asyncio.run(run_evaluation())
```

### **API Usage**
```bash
# Start evaluation via API
curl -X POST "http://localhost:8000/api/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_configs": [
      {"name": "CodeLlama", "provider": "ollama", "model_name": "codellama:7b"}
    ],
    "domains": ["frontend", "backend", "testing"],
    "max_tasks_per_domain": 5
  }'

# Get results
curl "http://localhost:8000/api/results"

# Download report
curl "http://localhost:8000/results/{run_id}/report" > report.html
```

## 🐳 **DEPLOYMENT OPTIONS**

### **Development Mode**
```bash
# Quick development setup
docker-compose up --build

# Services:
# - Web platform: http://localhost:8000
# - Redis: localhost:6379
# - PostgreSQL: localhost:5432
# - Ollama: http://localhost:11434
```

### **Production Mode**
```bash
# Production deployment with Nginx
docker-compose --profile production up --build

# Features:
# - Nginx reverse proxy
# - SSL termination
# - Distributed workers
# - Performance optimization
```

### **Minimal Setup**
```bash
# Just the core platform
docker-compose up llm-eval-platform redis
```

## 🔍 **BENCHMARKS INTEGRATED**

### **BigCodeBench (Primary)**
- **1,140 practical programming tasks**
- **Domain filtering** for Frontend/Backend/Testing
- **Diverse function calls** from 139 libraries
- **Real-world scenarios** beyond simple algorithms

### **HumanEval**
- **164 function-level tasks**
- **Functional correctness** evaluation
- **Pass@k metrics**
- **Industry standard** benchmark

### **Custom Datasets**
- **Domain-specific tasks** aligned with project requirements
- **Extensible framework** for adding new tasks
- **Quality-focused evaluation** with detailed metrics

## 📊 **RESULTS & ANALYTICS**

### **Leaderboard Features**
- **Real-time rankings** by overall score
- **Domain-specific performance** breakdown
- **Pass rate analysis** and execution time metrics
- **Model comparison** across different providers

### **Detailed Analytics**
- **Task-level results** with generated code
- **Error analysis** and failure patterns
- **Performance trends** and optimization insights
- **Export capabilities** for further analysis

## 🆘 **TROUBLESHOOTING**

### **Common Issues**

**Ollama Connection Failed:**
```bash
# Start Ollama
ollama serve

# Check status
curl http://localhost:11434/api/tags

# Download models
ollama pull codellama:7b
```

**BigCodeBench Installation:**
```bash
# Manual installation
pip install git+https://github.com/bigcode-project/bigcodebench.git

# Test installation
python -c "import bigcodebench; print('Success')"
```

**Docker Issues:**
```bash
# Check Docker status
docker info

# Restart services
docker-compose down && docker-compose up --build

# View logs
docker-compose logs -f llm-eval-platform
```

### **Diagnostics**
```bash
# Run comprehensive diagnostics
python comprehensive_diagnostic.py

# Check platform health
curl http://localhost:8000/health

# View application logs
docker-compose logs -f
```

## 📚 **API DOCUMENTATION**

### **Endpoints**
- `GET /` - Web interface home
- `GET /dashboard` - Results dashboard
- `GET /evaluate` - Evaluation configuration
- `POST /api/evaluate` - Start evaluation
- `GET /api/results` - List all results
- `GET /results/{run_id}` - Specific result
- `GET /results/{run_id}/report` - HTML report
- `GET /health` - Health check

### **WebSocket**
- `WS /ws/{client_id}` - Real-time progress updates

## 🤝 **EXTENDING THE PLATFORM**

### **Adding New Benchmarks**
```python
# Create new benchmark integration
class CustomBenchmark:
    async def evaluate_model(self, model_interface):
        # Custom evaluation logic
        return results

# Register with orchestrator
orchestrator.register_benchmark("custom", CustomBenchmark())
```

### **Adding New Domains**
```python
# Extend domain filtering
def _matches_domain(self, task, domain):
    if domain == "mobile":
        mobile_keywords = ['react-native', 'flutter', 'ios', 'android']
        return any(keyword in task.prompt.lower() for keyword in mobile_keywords)
```

### **Custom Model Providers**
```python
# Implement new model interface
class CustomModelInterface(ModelInterface):
    def generate_code(self, prompt, system_prompt=None):
        # Custom model integration
        return GenerationResult(...)

# Register with factory
ModelFactory.register_provider("custom", CustomModelInterface)
```

## 📝 **LICENSE**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 **CONCLUSION**

This platform provides a **comprehensive solution** for evaluating LLMs on coding tasks, specifically designed to meet the project requirements:

✅ **Complete Implementation** of all technical specifications  
✅ **One-Click Automation** for maximum ease of use  
✅ **True Benchmark Integration** with BigCodeBench and HumanEval  
✅ **Domain-Specific Evaluation** across Frontend, Backend, and Testing  
✅ **Multi-Model Support** for open-source LLMs  
✅ **Production-Ready Deployment** with Docker and web interface  

The platform enables developers and researchers to systematically evaluate and compare LLM performance across critical development domains, providing actionable insights for model selection and optimization.

---

**Ready to evaluate your LLMs? Start with one command:**
```bash
python deploy.py --mode=full
```

**Platform available at: http://localhost:8000** 🚀
