"""
Personalized Healthcare System - Startup Script
Starts both backend (FastAPI) and frontend (Streamlit) servers
"""
import os
import sys
import subprocess
import time
import signal
import platform

def check_port(port):
    """Check if a port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting Backend API Server (FastAPI)...")
    print("=" * 60)
    
    # Check if backend is already running
    if check_port(8000):
        print("⚠️  Backend server already running on port 8000")
        print("   If you need to restart, stop the existing process first")
        return None
    
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if platform.system() == 'Windows':
        backend_process = subprocess.Popen(
            [sys.executable, 'app.py'],
            cwd=backend_dir,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        backend_process = subprocess.Popen(
            [sys.executable, 'app.py'],
            cwd=backend_dir,
            preexec_fn=os.setsid
        )
    
    print("✅ Backend starting on http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    return backend_process

def start_frontend():
    """Start the Streamlit frontend server"""
    print("\n🌐 Starting Frontend UI (Streamlit)...")
    print("=" * 60)
    
    # Check if frontend is already running
    if check_port(8501):
        print("⚠️  Frontend server already running on port 8501")
        print("   If you need to restart, stop the existing process first")
        return None
    
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    if platform.system() == 'Windows':
        frontend_process = subprocess.Popen(
            ['streamlit', 'run', 'app.py', '--server.port=8501'],
            cwd=frontend_dir,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        frontend_process = subprocess.Popen(
            ['streamlit', 'run', 'app.py', '--server.port=8501'],
            cwd=frontend_dir,
            preexec_fn=os.setsid
        )
    
    print("✅ Frontend starting on http://localhost:8501")
    return frontend_process

def main():
    """Main startup function"""
    print("\n" + "=" * 60)
    print("🏥 PERSONALIZED HEALTHCARE SYSTEM")
    print("=" * 60)
    
    processes = []
    
    try:
        # Start backend
        backend_process = start_backend()
        if backend_process:
            processes.append(backend_process)
            time.sleep(3)  # Wait for backend to initialize
        
        # Start frontend
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(frontend_process)
            time.sleep(2)  # Wait for frontend to initialize
        
        if not processes:
            print("\n⚠️  No servers were started (both ports already in use)")
            print("   Run this script after stopping existing servers")
            return
        
        print("\n" + "=" * 60)
        print("✅ APPLICATION READY!")
        print("=" * 60)
        print("\n📍 Access the application:")
        print("   🌐 Frontend UI: http://localhost:8501")
        print("   🔧 Backend API: http://localhost:8000")
        print("   📖 API Docs: http://localhost:8000/docs")
        print("\n💡 Press Ctrl+C to stop all servers\n")
        print("=" * 60)
        
        # Keep the script running and monitor processes
        while True:
            time.sleep(1)
            # Check if any process has terminated
            for proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️  A server process terminated unexpectedly")
                    raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        for proc in processes:
            try:
                if platform.system() == 'Windows':
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception as e:
                print(f"   Error stopping process: {e}")
                try:
                    proc.kill()
                except:
                    pass
        
        print("✅ All servers stopped")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Clean up processes
        for proc in processes:
            try:
                proc.kill()
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()
