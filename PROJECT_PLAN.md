# 🚀 LLM Coding Benchmark Platform

A comprehensive platform for evaluating open-source LLMs on Frontend, Backend, and Testing domains.

## 🎯 Project Goals

Evaluate LLMs across 3 critical development domains:
- **Frontend**: React components, UI logic, styling, user interactions
- **Backend**: API endpoints, database operations, business logic, server functionality  
- **Testing**: Unit tests, integration tests, E2E testing scenarios

## 🏗️ Architecture

Built on **BigCodeBench** foundation with domain-specific extensions:

```
Platform Architecture:
├── BigCodeBench Core (Base evaluation engine)
├── Domain-Specific Datasets
│   ├── Frontend (React, Vue, Angular)
│   ├── Backend (APIs, databases, auth)
│   └── Testing (Unit, integration, E2E)
├── Multi-LLM Support (Ollama, OpenAI, Anthropic)
├── Web UI (One-click evaluation)
└── Results Dashboard (Leaderboards, analytics)
```

## 🛠️ Implementation Plan

### Phase 1: Foundation Integration
1. ✅ Integrate BigCodeBench as evaluation engine
2. ✅ Configure multi-LLM support
3. ✅ Set up Docker environment

### Phase 2: Domain-Specific Datasets  
1. ✅ Frontend dataset (React components, UI interactions)
2. ✅ Backend dataset (REST APIs, database operations)
3. ✅ Testing dataset (Test generation, automation)

### Phase 3: Web Interface
1. ✅ One-click evaluation interface
2. ✅ Real-time progress tracking
3. ✅ Results dashboard with leaderboards

### Phase 4: Advanced Features
1. 🔄 Additional benchmark integrations (HumanEval, EvalPlus)
2. 🔄 Custom dataset upload
3. 🔄 Comparative analysis tools

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository>
cd llm-coding-benchmark
docker-compose up --build

# Access web interface
open http://localhost:8000
```

## 📊 Supported Benchmarks

- **BigCodeBench**: Primary evaluation framework
- **HumanEval**: Function-level code generation
- **EvalPlus**: Extended test cases
- **Custom Datasets**: Domain-specific tasks

## 🤖 Supported Models

- **Local**: Ollama (CodeLlama, DeepSeek, Qwen2.5-Coder)
- **API**: OpenAI, Anthropic, HuggingFace
- **Custom**: Any model via API interface
