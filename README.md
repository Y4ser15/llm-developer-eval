# 🚀 LLM Coding Evaluation Platform

A benchmark for evaluating open-source Large Language Models on code generation across Frontend, Backend, and Testing domains.

## Features

- **🤖 Multi-Model Support**: Ollama local models, OpenAI, Anthropic, HuggingFace APIs
- **🔒 Secure Execution**: E2B sandbox for safe code evaluation  
- **📊 Real-Time Dashboard**: Live progress updates and results visualization
- **🎯 Domain-Specific**: Frontend (React), Backend (APIs), Testing (Unit/E2E)
- **🏆 Standard Benchmarks**: BigCodeBench (1,140 tasks) + HumanEval (164 tasks)
- **⚡ One-Click Automation**: Start evaluation and get comprehensive results

## Tech Stack

- **Backend**: FastAPI + Uvicorn (async web framework)
- **Models**: Ollama, OpenAI API, Anthropic API, HuggingFace API
- **Execution**: E2B sandbox (secure cloud code execution)
- **Benchmarks**: BigCodeBench, HumanEval with domain filtering
- **Storage**: Redis (caching) + JSON (results)
- **Frontend**: Jinja2 templates + JavaScript + WebSockets
- **Deployment**: Docker Compose

## Quick Start

### Prerequisites
- Docker Desktop
- E2B API key

### Deploy
```bash
# Clone and navigate to project
cd llm-developer-eval

# Add your E2B API key to .env
echo "E2B_API_KEY=e2b_your_key_here" >> .env

# Start all services
docker-compose up --build

# Access platform
open http://localhost:8000
```

### Development Mode
```bash
pip install -r requirements.txt
python app.py
# Access: http://localhost:8000
```

## Usage

1. **Configure Models**: Add API keys or use local Ollama models
2. **Select Domains**: Choose Frontend, Backend, and/or Testing evaluation
3. **Start Evaluation**: One-click automated benchmark execution
4. **Monitor Progress**: Real-time WebSocket updates
5. **View Results**: Interactive dashboard with model comparisons
6. **Export Data**: JSON/HTML reports for analysis

## Current Evaluation

### Domains
- **Frontend**: React components, UI logic, styling, user interactions
- **Backend**: API endpoints, database operations, business logic, server-side functionality  
- **Testing**: Unit tests, integration tests, end-to-end testing scenarios

### Metrics
- **Functional Correctness**: Pass@k test success rate
- **Code Quality**: Syntax validation and execution success
- **Domain Performance**: Task completion across Frontend/Backend/Testing
- **Execution Time**: Performance and resource usage

### Supported Models
- **Local**: CodeLlama, DeepSeek Coder, Qwen2.5-Coder (via Ollama)
- **API**: GPT-4, Claude 3, HuggingFace models
- **Custom**: vLLM servers and custom endpoints

## Future Work

### Advanced Metrics
- **Code Quality**: Lint analysis (Pylint, Prettier), Code BLEU, AST validation
- **Security**: SonarQube SAST, Bandit scanning, vulnerability detection (CWE)
- **Performance**: EffiBench integration, memory/CPU analysis
- **Accessibility**: WCAG 2.2 compliance testing (Wave, axe DevTools, Pa11y)

### Enhanced Evaluation
- **LLM as Judge**: Model-based code quality assessment
- **Coverage@n**: Success rate within n attempts  
- **Advanced Benchmarks**: SWE-bench, LiveCodeBench, Spider 2.0
- **Judge0 Integration**: Alternative execution environment

### Platform Improvements
- **Extended Datasets**: Web-bench, Fullstack-bench integration
- **Advanced Analytics**: Detailed performance comparisons
- **CI/CD Integration**: Automated evaluation pipelines

## Architecture

```
Web Interface (FastAPI) → Model APIs → E2B Sandbox → Results Dashboard
     ↓                        ↓            ↓              ↓
WebSocket Updates      Code Generation  Safe Execution  Real-time UI
```

## Project Structure

```
llm-developer-eval/
├── app.py                    # Main entry point
├── docker-compose.yml        # Service orchestration  
├── requirements.txt          # Python dependencies
├── src/
│   ├── core/                 # Model interfaces & datasets
│   ├── evaluation/           # Evaluation engine
│   ├── utils/                # Report generation
│   └── web/                  # FastAPI application
├── datasets/                 # Domain-specific tasks
└── results/                  # Evaluation outputs
```

## Configuration

### Environment Variables (.env)
```bash
# Required for code execution
E2B_API_KEY=e2b_your_key_here

# Optional API keys for cloud models
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_hf_key

# Local model server
OLLAMA_BASE_URL=http://localhost:11434
```

### Ollama Setup (Optional)
```bash
# Install recommended models
ollama pull codellama:7b
ollama pull deepseek-coder:6.7b
ollama pull qwen2.5-coder:7b
```

## Services

- **Main Platform**: http://localhost:8000
- **Dashboard**: http://localhost:8000/dashboard  
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Stop services: `docker-compose down`

---

**Ready to benchmark LLMs?** Start with `docker-compose up --build` and visit http://localhost:8000
