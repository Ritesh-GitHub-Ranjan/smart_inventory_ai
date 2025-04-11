# utils/ollama_manager.py
import subprocess
import psutil

def is_ollama_running(port=11434):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def start_ollama_server():
    if not is_ollama_running():
        subprocess.Popen(['ollama', 'serve'], creationflags=subprocess.CREATE_NO_WINDOW)
        print(" Ollama server started.")
    else:
        print(" Ollama already running.")
