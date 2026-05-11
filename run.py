#!/usr/bin/env python3
"""
SparkLearn: Ignite Your Learning — Quick Start
"""
import sys, os, subprocess

def check_python():
    if sys.version_info < (3, 8):
        print(" Python 3.8+ required"); sys.exit(1)
    print(f" Python {sys.version.split()[0]}")

def check_env():
    from pathlib import Path
    if not Path('.env').exists():
        if Path('.env.example').exists():
            import shutil; shutil.copy('.env.example', '.env')
            print("📋 Created .env from .env.example")
            print("   → Add GEMINI_API_KEY (fallback) or run: ollama pull qwen2:0.5b")
        else:
            print("  No .env found")

def install_deps():
    print(" Checking dependencies...")
    missing = []
    for pkg in ['flask','cv2','numpy','sklearn']:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"📥 Installing missing packages...")
        subprocess.check_call([sys.executable,'-m','pip','install','-r','requirements.txt','-q'])
    print(" Dependencies ready")

def check_ollama():
    try:
        import urllib.request
        urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)
        print(" Ollama running (primary AI backend)")
    except:
        print("  Ollama not running — falling back to Gemini")
        print("   To use Ollama: ollama serve  (in another terminal)")
        print("   Pull model:    ollama pull qwen2:0.5b")

def main():
    print("\n" + "═"*56)
    print("   SparkLearn: Ignite Your Learning")
    print("   AI Placement Preparation System")
    print("═"*56)
    check_python(); check_env(); install_deps(); check_ollama()
    port = int(os.getenv('PORT', 5000))
    print(f"\n  http://localhost:{port}")
    print("    Ctrl+C to stop\n")
    from app import app, _build_vectors
    _build_vectors()
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()
