#!/usr/bin/env python3
"""
Fix Login Issues - Helper Script for Friends
Diagnoses and fixes common login problems when pulling the project
"""
import subprocess
import sys
import time
import requests
import os
import json
from pathlib import Path

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",  # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m",   # Red
        "reset": "\033[0m"     # Reset
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    print(f"{colors.get(status, '')}{icons.get(status, '')} {message}{colors['reset']}")

def check_python_version():
    """Check if Python version is compatible"""
    print_status("Checking Python version...", "info")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Incompatible (need 3.8+)", "error")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print_status("Checking dependencies...", "info")
    
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "requests", 
        "pandas", "numpy", "scikit-learn", "joblib", "bcrypt"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        print_status("All dependencies installed", "success")
        return True
    else:
        print_status(f"Missing packages: {', '.join(missing)}", "error")
        return False

def install_dependencies():
    """Install missing dependencies"""
    print_status("Installing dependencies...", "info")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print_status("Dependencies installed successfully", "success")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install dependencies: {e}", "error")
        return False

def check_file_structure():
    """Check if project files are present"""
    print_status("Checking project structure...", "info")
    
    required_files = [
        "backend/app.py",
        "backend/auth/models.py", 
        "backend/auth/routes.py",
        "frontend/app.py",
        "requirements.txt"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if not missing:
        print_status("All required files present", "success")
        return True
    else:
        print_status(f"Missing files: {', '.join(missing)}", "error")
        print_status("Make sure you're in the project root directory", "warning")
        return False

def kill_existing_processes():
    """Kill any existing processes on ports 8000 and 8501"""
    print_status("Checking for existing processes on ports 8000 and 8501...", "info")
    
    import psutil
    
    for port in [8000, 8501]:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                if connections:
                    for conn in connections:
                        if hasattr(conn, 'laddr') and conn.laddr.port == port:
                            print_status(f"Killing process {proc.info['pid']} using port {port}", "warning")
                            proc.terminate()
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

def start_backend():
    """Start the backend server"""
    print_status("Starting backend server...", "info")
    
    try:
        # Start backend in background
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend.app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for backend to start
        for i in range(10):
            try:
                response = requests.get("http://localhost:8000/api/health", timeout=2)
                if response.status_code == 200:
                    print_status("Backend started successfully", "success")
                    print_status("Backend available at: http://localhost:8000", "info")
                    return backend_process
            except:
                pass
            time.sleep(1)
        
        print_status("Backend failed to start within 10 seconds", "error")
        backend_process.terminate()
        return None
        
    except Exception as e:
        print_status(f"Failed to start backend: {e}", "error")
        return None

def test_authentication():
    """Test if authentication endpoints work"""
    print_status("Testing authentication...", "info")
    
    try:
        # Test login with default admin user
        login_data = {
            "email": "admin@healthcare.com",
            "password": "Admin123!"
        }
        
        response = requests.post(
            "http://localhost:8000/api/auth/login",
            json=login_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print_status("Authentication working correctly", "success")
            print_status(f"Test login successful for: {result.get('user', {}).get('email')}", "success")
            return True
        else:
            print_status(f"Authentication failed with status: {response.status_code}", "error")
            return False
            
    except Exception as e:
        print_status(f"Authentication test failed: {e}", "error")
        return False

def start_frontend():
    """Start the frontend server"""
    print_status("Starting frontend server...", "info")
    
    try:
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", 
            "run", "frontend/app.py", 
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        
        print_status("Frontend started successfully", "success") 
        print_status("Frontend available at: http://localhost:8501", "info")
        return frontend_process
        
    except Exception as e:
        print_status(f"Failed to start frontend: {e}", "error")
        return None

def show_login_instructions():
    """Show login instructions for friends"""
    print("\n" + "="*60)
    print_status("🎉 SETUP COMPLETE - LOGIN INSTRUCTIONS", "success")
    print("="*60)
    
    print_status("1. Open your browser and go to: http://localhost:8501", "info")
    print_status("2. You'll see a login page", "info") 
    print_status("3. Use these test accounts:", "info")
    
    print("\n📋 TEST ACCOUNTS:")
    print("   👤 Admin Account:")
    print("      Email: admin@healthcare.com")
    print("      Password: Admin123!")
    print()
    print("   👨‍⚕️ Doctor Account:")
    print("      Email: doctor@healthcare.com") 
    print("      Password: Doctor123!")
    print()
    print("   🏥 Patient Account:")
    print("      Email: patient@healthcare.com")
    print("      Password: Patient123!")
    
    print("\n🔗 USEFUL LINKS:")
    print("   • Frontend: http://localhost:8501")
    print("   • Backend API: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    
    print("\n⚠️  IMPORTANT:")
    print("   • Keep this terminal window open")
    print("   • Both frontend and backend are now running")
    print("   • Press Ctrl+C to stop both services")

def main():
    """Main diagnosis and fix routine"""
    print("🏥 HEALTHCARE APP - LOGIN FIX TOOL")
    print("="*60)
    print_status("This tool will diagnose and fix login issues", "info")
    print_status("Make sure you're in the project root directory", "warning")
    print()
    
    # Step 1: Check Python version
    if not check_python_version():
        print_status("Please install Python 3.8 or higher", "error")
        return False
    
    # Step 2: Check file structure
    if not check_file_structure():
        print_status("Please navigate to the correct project directory", "error")
        return False
    
    # Step 3: Check dependencies
    if not check_dependencies():
        print_status("Installing missing dependencies...", "warning")
        if not install_dependencies():
            print_status("Please run manually: pip install -r requirements.txt", "error")
            return False
    
    # Step 4: Kill existing processes (optional)
    try:
        kill_existing_processes()
    except ImportError:
        print_status("psutil not available - skipping process cleanup", "warning")
    except Exception as e:
        print_status(f"Process cleanup failed: {e}", "warning")
    
    # Step 5: Start backend
    backend_process = start_backend()
    if not backend_process:
        print_status("Cannot start without backend - check for errors above", "error")
        return False
    
    # Step 6: Test authentication
    if not test_authentication():
        print_status("Authentication test failed - backend might be starting", "warning")
        print_status("Continuing anyway...", "info")
    
    # Step 7: Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print_status("Frontend failed to start", "error")
        backend_process.terminate()
        return False
    
    # Step 8: Show instructions
    show_login_instructions()
    
    # Keep running
    try:
        print_status("\nPress Ctrl+C to stop both services", "info")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print_status("\nStopping services...", "info")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print_status("Services stopped", "success")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print_status("\n❌ Setup failed - check errors above", "error")
        input("Press Enter to exit...")
    else:
        print_status("\n✅ Setup completed successfully!", "success")