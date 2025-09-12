#!/usr/bin/env python3
"""
Healthcare System Startup Script
Start both backend and frontend services
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def start_backend():
    """Start the backend server"""
    print("🚀 Starting Healthcare Backend Server...")
    print("📍 Available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("-" * 50)
    
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    return subprocess.Popen(backend_cmd)

def start_frontend():
    """Start the frontend server"""
    print("🎨 Starting Healthcare Frontend...")
    print("📍 Available at: http://localhost:8501")
    print("-" * 50)
    
    frontend_cmd = [
        sys.executable, "-m", "streamlit",
        "run", "frontend/app.py",
        "--server.port", "8501"
    ]
    
    return subprocess.Popen(frontend_cmd)

def main():
    """Main startup function"""
    print("🏥 Healthcare System Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend/app.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    try:
        choice = input("Choose startup mode:\n1. Backend only\n2. Frontend only\n3. Both (recommended)\nEnter choice (1-3): ").strip()
        
        processes = []
        
        if choice in ["1", "3"]:
            backend_process = start_backend()
            processes.append(("Backend", backend_process))
            if choice == "3":
                time.sleep(3)  # Wait for backend to start
        
        if choice in ["2", "3"]:
            frontend_process = start_frontend()
            processes.append(("Frontend", frontend_process))
        
        if not processes:
            print("❌ Invalid choice. Exiting.")
            return
        
        print("\n✅ Services started successfully!")
        print("Press Ctrl+C to stop all services")
        
        # Wait for processes
        try:
            for name, process in processes:
                process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            for name, process in processes:
                process.terminate()
                print(f"✅ {name} stopped")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()