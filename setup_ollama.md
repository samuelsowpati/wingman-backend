# 🦙 Ollama Setup Guide for Air Force RAG Chatbot

This guide will help you install and configure Ollama with Llama 3.1 8B for your Air Force chatbot.

## 📦 Step 1: Install Python Dependencies

```bash
cd D:\Codes\wingman2
venv\Scripts\activate
pip install -r requirements.txt
```

## 🦙 Step 2: Install Ollama

### **Option A: Windows Installer (Recommended)**
1. Download Ollama for Windows: https://ollama.com/download/windows
2. Run the installer (`OllamaSetup.exe`)
3. Follow the installation wizard

### **Option B: Manual Installation**
```powershell
# Using winget (if available)
winget install Ollama.Ollama

# Or using Chocolatey
choco install ollama
```

## 🚀 Step 3: Download Llama 3.1 8B Model

Open **Command Prompt** or **PowerShell** and run:

```bash
# This will download ~4.7GB model
ollama pull llama3.2:3b
```

**Download time:** ~10-30 minutes depending on internet speed

## ✅ Step 4: Verify Installation

```bash
# Check if Ollama is running
ollama list

# Should show:
# NAME            ID              SIZE     MODIFIED
# llama3.2:3b     42182c40c896    4.7 GB   X minutes ago

# Test the model
ollama run llama3.2:3b "Hello, I'm testing Ollama integration"
```

## 🔧 Step 5: Start Ollama Server

Ollama typically starts automatically, but if needed:

```bash
# Start Ollama server (usually runs on http://localhost:11434)
ollama serve
```

## 🧪 Step 6: Test Integration

1. **Start your Air Force API:**
```bash
cd D:\Codes\wingman2
venv\Scripts\activate
$env:PINECONE_API_KEY="your_key_here"
python main.py
```

2. **Test LLM connectivity:**
```bash
# In a separate terminal
cd D:\Codes\wingman2
venv\Scripts\activate
python -c "import asyncio; from llm_service import test_ollama_connection; asyncio.run(test_ollama_connection())"
```

3. **Test via API:**
   - Go to `http://localhost:8000/docs`
   - Use `POST /api/chat` with: `{"message": "What does the AFOSI commander do?", "history": []}`

## 📊 Expected Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **Model Loading** | 5-15s | First request only |
| **Response Generation** | 3-10s | Depends on complexity |
| **Memory Usage** | ~6GB RAM | For 8B model |

## 🔧 Troubleshooting

### **Ollama Not Starting:**
```bash
# Check if port 11434 is in use
netstat -ano | findstr :11434

# Kill existing process if needed
taskkill /PID <process_id> /F

# Restart Ollama
ollama serve
```

### **Model Not Found:**
```bash
# Re-download the model
ollama rm llama3.2:3b
ollama pull llama3.2:3b
```

### **Memory Issues:**
If you have <8GB RAM, try smaller models:
```bash
# Download smaller model (2.7GB)
ollama pull llama3.1:7b

# Update wingman2/llm_service.py line 26:
# self.model = "llama3.1:7b"
```

## 🎯 What You'll Get

**Before (Raw Text):**
```
Based on Air Force documentation, here's what I found regarding: What does the AFOSI commander do?

Source 1 (AFI - https://static.e-publishing.af.mil/...):
ROLES AND RESPONSIBILITIES 2.1. Air Force Office of Special Investigations (AFOSI). 2.2. The AFOSI Commander will: ...
```

**After (LLM-Generated):**
```
Based on the Air Force Instruction 71-101, the AFOSI Commander has several key responsibilities:

**Primary Role:**
The AFOSI Commander directs all Air Force special investigations and counterintelligence operations...

**Specific Responsibilities:**
1. Strategic oversight of investigation priorities
2. Coordination with DoD and civilian law enforcement
3. Resource allocation for special investigations
4. Policy implementation for counterintelligence operations

**Authority:**
The commander has direct authority over...

*Based on AFI 71-101v3, Chapter 2*
```

## 🚀 Ready to Test!

Your system now has:
- ✅ **Vector Search** (Pinecone + Sentence Transformers)
- ✅ **LLM Generation** (Ollama + Llama 3.1 8B)  
- ✅ **Fallback System** (Works even if LLM is down)
- ✅ **Source Attribution** (Shows document sources)

The complete RAG pipeline is ready! 🎉