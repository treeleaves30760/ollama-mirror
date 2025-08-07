# 🪟 Windows Usage Guide - Ollama Mirror Server

## ✅ **FIXED AND READY TO USE!**

All dependency conflicts have been resolved and the server is working perfectly on Windows.

## 🚀 **Quick Start**

### **Option 1: Easy Start (Recommended)**
Simply **double-click `start_server.bat`** and you're done!

### **Option 2: PowerShell**
```powershell
.\start_server.ps1
```

### **Option 3: Manual**
```powershell
python cli.py start
```

## 🧪 **Test the Server**

**Double-click `test_server.bat`** or run:
```powershell
.\test_server.ps1
```

## 🔧 **Using with Ollama**

### **Set Environment Variable (PowerShell):**
```powershell
# For current session
$env:OLLAMA_HOST = "http://localhost:11434"

# Make it permanent
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "http://localhost:11434", "User")
```

### **Set Environment Variable (Command Prompt):**
```cmd
set OLLAMA_HOST=http://localhost:11434
```

### **Test Model Pull:**
```powershell
# After setting OLLAMA_HOST
ollama pull llama3.2:1b
```

## 📊 **Monitor the Server**

### **Check Server Status:**
```powershell
curl http://localhost:11434/
```

### **View Cache Statistics:**
```powershell
curl http://localhost:11434/mirror/stats
```

### **Health Check:**
```powershell
curl http://localhost:11434/mirror/health
```

## 🔧 **CLI Commands**

```powershell
# Start server
python cli.py start

# Check health
python cli.py health

# View cache status
python cli.py cache status

# Generate config
python cli.py config generate

# Clear cache
python cli.py cache clear
```

## 📁 **File Structure**

```
ollama-mirror/
├── start_server.bat         # 🖱️ Double-click to start!
├── start_server.ps1         # PowerShell version
├── test_server.bat          # 🖱️ Double-click to test!
├── test_server.ps1          # PowerShell test
├── ollama_mirror.py         # Main server
├── cli.py                   # Command line tool
├── requirements.txt         # ✅ Updated dependencies
├── README.md                # Full documentation
└── cache/                   # Model cache (auto-created)
```

## ⚠️ **Important Notes**

1. **The server automatically stops existing Ollama processes** to avoid port conflicts
2. **All dependencies are automatically installed** when using the batch files
3. **Cache directory is created automatically**
4. **Compatible with ollama 0.4.7** and all dependencies are resolved

## 🎉 **What's Working**

✅ All endpoint tests pass  
✅ Health checks work  
✅ Mirror statistics work  
✅ Model caching ready  
✅ No dependency conflicts  
✅ Windows batch files for easy use  
✅ PowerShell scripts  
✅ Compatible with latest FastAPI/Pydantic  

## 💡 **Quick Usage Flow**

1. **Double-click `start_server.bat`**
2. Wait for "Server is running" message
3. Open new PowerShell window
4. Run: `$env:OLLAMA_HOST = "http://localhost:11434"`
5. Test: `ollama pull llama3.2:1b`
6. **Done!** Models will be cached locally

## 🐛 **Troubleshooting**

### **PowerShell Execution Policy Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Port in Use:**
The batch file automatically stops existing Ollama processes

### **Dependencies:**
Run the batch file - it automatically installs everything

### **Testing:**
```powershell
python test_mirror.py
```

---

🎉 **The Ollama Mirror Server is now fully functional on Windows!**
