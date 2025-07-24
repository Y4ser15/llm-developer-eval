# 🚀 LLM Coding Evaluation Platform - Quick Start

## **ONE-CLICK DEPLOYMENT**

```bash
# Quick deployment (recommended)
python deploy.py

# Platform available at: http://localhost:8000
```

## **PROJECT OVERVIEW**

A comprehensive evaluation platform for Large Language Models (LLMs) that systematically benchmarks coding capabilities across **Frontend**, **Backend**, and **Testing** domains.

## **TECH STACK**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI + Uvicorn | High-performance async API server |
| **Models** | Ollama, OpenAI, Anthropic, HF | Multi-provider LLM support |
| **Evaluation** | BigCodeBench, HumanEval | Industry-standard benchmarks |
| **Database** | Redis + PostgreSQL | Caching and results storage |
| **Frontend** | Jinja2 + JavaScript | Interactive web interface |
| **Deployment** | Docker + Docker Compose | Containerized microservices |

## **DEPLOYMENT OPTIONS**

### **Option 1: Automated Setup (Recommended)**
```bash
# One-click deployment with all services
python deploy.py --mode docker

# Production deployment with Nginx
python deploy.py --mode production

# Native Python deployment
python deploy.py --mode native
```

### **Option 2: Manual Docker Setup**
```bash
# Quick start
docker-compose up --build

# Production mode
docker-compose --profile production up --build

# Minimal setup (core services only)
docker-compose up llm-eval-platform redis
```

### **Option 3: Development Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Start application
python app.py --reload

# Access: http://localhost:8000
```

## **CONFIGURATION**

### **API Keys (.env file)**
```bash
# Optional: Add API keys for cloud models
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key
```

### **Ollama Models (Local LLMs)**
```bash
# Install recommended models
python deploy.py --setup-ollama

# Or manually:
ollama pull codellama:7b
ollama pull deepseek-coder:6.7b
ollama pull qwen2.5-coder:7b
```

## **USAGE WORKFLOW**

1. **🌐 Access Platform**: `http://localhost:8000`
2. **⚙️ Configure Models**: Add API keys or use local Ollama models
3. **🎯 Start Evaluation**: Go to `/evaluate` and select models/domains
4. **📊 Monitor Progress**: Real-time updates via WebSocket
5. **📈 View Results**: Interactive dashboard at `/dashboard`
6. **📄 Export Reports**: JSON, CSV, HTML formats available

## **KEY FEATURES**

- **🚀 One-Click Evaluation**: Fully automated with progress tracking
- **🤖 Multi-Model Support**: Ollama, OpenAI, Anthropic, HuggingFace
- **🏆 Benchmark Integration**: BigCodeBench (1,140 tasks), HumanEval (164 tasks)
- **🎯 Domain-Specific**: Frontend, Backend, Testing evaluation
- **📊 Real-Time Dashboard**: Interactive leaderboards and analytics
- **🐳 Container-Ready**: Docker deployment with microservices

## **ARCHITECTURE**

```
Platform Architecture:
├── 🌐 Web Interface (FastAPI + WebSocket)
├── 🤖 Model Interfaces (Multi-provider support)  
├── 🧪 Evaluation Engine (BigCodeBench + HumanEval)
├── 📊 Results Dashboard (Real-time analytics)
├── 🐳 Containerized Services (Docker Compose)
└── 📈 Report Generation (HTML/JSON/CSV)
```

## **SERVICE ENDPOINTS**

- **🏠 Home**: `http://localhost:8000/`
- **⚙️ Evaluation**: `http://localhost:8000/evaluate`
- **📊 Dashboard**: `http://localhost:8000/dashboard`
- **🩺 Health Check**: `http://localhost:8000/health`
- **📚 API Docs**: `http://localhost:8000/docs`

## **EVALUATION DOMAINS**

### **🎨 Frontend Development**
- React component development
- UI logic and state management
- CSS styling and responsiveness
- User interaction handling
- Accessibility compliance

### **⚙️ Backend Development**
- REST API development
- Database operations and ORM
- Authentication and authorization
- Business logic implementation
- Server-side architecture

### **🧪 Testing & QA**
- Unit test generation
- Integration test suites
- End-to-end automation
- Mock data generation
- Test coverage analysis

## **SUPPORTED MODELS**

### **Local Models (Ollama)**
- CodeLlama 7B/13B/70B
- DeepSeek Coder 6.7B/33B
- Qwen2.5 Coder 7B/32B
- StarCoder2 7B/15B
- CodeGemma 7B

### **API Models**
- OpenAI: GPT-4, GPT-3.5-Turbo
- Anthropic: Claude 3 (Opus/Sonnet/Haiku)
- HuggingFace: CodeLlama, WizardCoder, StarCoder

### **Custom Servers**
- vLLM inference servers
- OpenAI-compatible APIs
- Custom model endpoints

## **TROUBLESHOOTING**

### **Common Issues**

**🔧 Ollama Connection**
```bash
# Start Ollama
ollama serve

# Check models
ollama list
```

**🔧 Docker Issues**
```bash
# Restart services
docker-compose down && docker-compose up --build

# View logs
docker-compose logs -f
```

**🔧 BigCodeBench Setup**
```bash
# Manual installation
pip install git+https://github.com/bigcode-project/bigcodebench.git
```

## **RESULTS & ANALYTICS**

### **Sample Leaderboard**
```
🏆 Model Performance:
1. GPT-4 Turbo: 0.847 (Frontend: 0.891, Backend: 0.823, Testing: 0.827)
2. Claude 3.5 Sonnet: 0.834 (Frontend: 0.856, Backend: 0.812, Testing: 0.834) 
3. CodeLlama 70B: 0.782 (Frontend: 0.743, Backend: 0.798, Testing: 0.805)
4. DeepSeek Coder 33B: 0.761 (Frontend: 0.721, Backend: 0.789, Testing: 0.773)
```

### **Export Formats**
- **JSON**: Programmatic access and analysis
- **CSV**: Data analysis and visualization
- **HTML**: Comprehensive interactive reports

**Platform available at: http://localhost:8000** ✨
