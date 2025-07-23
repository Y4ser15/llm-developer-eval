# 🚀 Quick Local Development Setup

## **Option 1: Super Simple (Recommended for Testing)**

```bash
# 1. Install basic dependencies
python dev_server.py --install-deps

# 2. Set up environment
python dev_server.py --setup

# 3. Run quick test
python dev_server.py --test

# 4. Start development server
python dev_server.py
```

**That's it! Open http://localhost:8000** 🎉

---

## **Option 2: Manual Setup**

### **Step 1: Environment Setup**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file (optional - only needed for API models)
# Add your API keys if you want to test cloud models
```

### **Step 2: Install Dependencies**
```bash
# Install core dependencies
pip install fastapi uvicorn jinja2 requests python-dotenv pydantic

# OR install full requirements (includes BigCodeBench)
pip install -r requirements.txt
```

### **Step 3: Start Server**
```bash
# Start development server
python -m uvicorn src.web.comprehensive_app:app --reload --host localhost --port 8000
```

---

## **🔧 Environment File (.env)**

**Keep `.env.example` as template, create your own `.env`:**

```bash
# Your .env file (edit as needed)
PYTHONPATH=.
LOG_LEVEL=INFO
HOST=localhost
PORT=8000
DEBUG=true

# Optional API keys (only if you want to test cloud models)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Ollama (optional - for local models)
OLLAMA_BASE_URL=http://localhost:11434
```

---

## **🤖 Optional: Ollama Setup (Local Models)**

If you want to test with local models:

```bash
# 1. Install Ollama from https://ollama.com
# 2. Start Ollama
ollama serve

# 3. Download a coding model
ollama pull codellama:7b
```

---

## **🧪 Testing the Setup**

```bash
# Test platform components
python dev_server.py --test

# Check if web interface works
curl http://localhost:8000/health
```

---

## **📱 Accessing the Platform**

Once server is running:
- **Web Interface**: http://localhost:8000
- **Dashboard**: http://localhost:8000/dashboard  
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## **🔍 Troubleshooting**

**Port already in use:**
```bash
python dev_server.py --port 8001
```

**Dependencies missing:**
```bash
python dev_server.py --install-deps
```

**Import errors:**
```bash
# Make sure you're in the project directory
cd llm-developer-eval
export PYTHONPATH=.
python dev_server.py
```

**Platform test fails:**
```bash
# Install full requirements
pip install -r requirements.txt
python dev_server.py --test
```
