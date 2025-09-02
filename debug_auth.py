#!/usr/bin/env python3
"""
Authentication Debug Script
Helps diagnose and fix login issues when friends pull the project
"""
import requests
import json
import sys
import os
import subprocess
import time
from typing import List, Dict, Tuple, Any, Union

def check_backend_status() -> bool:
    """Check if backend is running and accessible"""
    print("Checking Backend Status...")
    print("-" * 40)
    
    try:
        # Test basic connectivity
        response = requests.get('http://localhost:8000/api/health', timeout=3)
        health_data = response.json()
        print("✓ Backend is running")
        print(f"   Status: {health_data.get('status')}")
        print(f"   Model loaded: {health_data.get('model_loaded', 'unknown')}")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Backend is NOT running")
        print("   Start it with: python start.py")
        return False
    except Exception as e:
        print(f"✗ Backend connection error: {e}")
        return False

def check_auth_endpoints() -> List[str]:
    """Check if authentication endpoints are available"""
    print("\nChecking Authentication Endpoints...")
    print("-" * 40)
    
    endpoints = [
        "/api/auth/login",
        "/api/auth/me", 
        "/api/auth/register"
    ]
    
    available_endpoints = []
    
    for endpoint in endpoints:
        try:
            url = f"http://localhost:8000{endpoint}"
            # Use HEAD request to check if endpoint exists
            response = requests.post(url, json={}, timeout=2)
            if response.status_code != 404:
                available_endpoints.append(endpoint)
                print(f"✓ {endpoint} - Available")
            else:
                print(f"✗ {endpoint} - Not Found")
        except Exception as e:
            print(f"! {endpoint} - Error: {str(e)[:50]}")
    
    return available_endpoints

def test_default_users() -> List[Dict[str, str]]:
    """Test login with default users"""
    print("\nTesting Default User Accounts...")
    print("-" * 40)
    
    default_users = [
        {"email": "admin@healthcare.com", "password": "Admin123!", "role": "admin"},
        {"email": "doctor@healthcare.com", "password": "Doctor123!", "role": "doctor"},
        {"email": "patient@healthcare.com", "password": "Patient123!", "role": "patient"}
    ]
    
    working_users = []
    
    for user in default_users:
        try:
            login_data = {
                "email": user["email"],
                "password": user["password"]
            }
            
            response = requests.post(
                "http://localhost:8000/api/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ {user['role'].upper()} login successful")
                print(f"   Email: {user['email']}")
                print(f"   Password: {user['password']}")
                print(f"   Token: {result.get('token', {}).get('access_token', 'N/A')[:20]}...")
                working_users.append(user)
            else:
                print(f"✗ {user['role'].upper()} login failed")
                print(f"   Status: {response.status_code}")
                print(f"   Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"✗ {user['role'].upper()} login error: {e}")
    
    return working_users

def check_user_store() -> bool:
    """Check if user store is properly initialized"""
    print("\nChecking User Store Initialization...")
    print("-" * 40)
    
    try:
        sys.path.append('backend')
        # Use absolute import to avoid module resolution issues
        from backend.auth.models import user_store
        
        print(f"✓ User store loaded")
        print(f"   Total users: {len(user_store.users)}")
        
        for user_id, user in user_store.users.items():
            print(f"   - {user.email} ({user.role}) - Active: {user.is_active}")
        
        return True
    except ImportError as ie:
        print(f"✗ Import error: {ie}")
        print("   Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"✗ User store error: {e}")
        return False

def create_startup_guide() -> None:
    """Create a startup guide for friends"""
    print("\nCreating Startup Guide...")
    print("-" * 40)
    
    # Create a simplified guide without problematic Unicode characters for better Windows compatibility
    guide = """
# Healthcare App - Quick Start Guide for Friends

## Step 1: Setup Environment
```bash
# Navigate to project directory
cd personalized-healthcare

# Run setup script (Windows)
.\\setup.bat

# OR manually create virtual environment
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

## Step 2: Start the Application
```bash
# Option A: Use startup script (recommended)
python start.py
# Then select option 3 (Both backend and frontend)

# Option B: Manual startup
# Terminal 1 - Backend:
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend:
python -m streamlit run frontend/app.py --server.port 8501
```

## Step 3: Access the Application
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend**: http://localhost:8501

## Step 4: Test Login
Use these default accounts:

### Admin Account
- **Email**: admin@healthcare.com
- **Password**: Admin123!

### Doctor Account  
- **Email**: doctor@healthcare.com
- **Password**: Doctor123!

### Patient Account
- **Email**: patient@healthcare.com
- **Password**: Patient123!

## Troubleshooting

### "Not Found" Error Solutions:

1. **Backend not running**: Make sure Step 2 completed successfully
2. **Wrong URL**: Check you're using http://localhost:8000 (not 8001 or other ports)
3. **Port conflicts**: Try different ports if 8000 is taken
4. **Firewall**: Allow Python through Windows firewall

### Quick Health Check:
```bash
# Test if backend is responding
curl http://localhost:8000/api/health

# Test login endpoint
curl -X POST http://localhost:8000/api/auth/login \\
  -H "Content-Type: application/json" \\
  -d "{\\"email\\":\\"admin@healthcare.com\\",\\"password\\":\\"Admin123!\\"}"
```

### If Still Having Issues:
1. Check Windows Defender/Antivirus isn't blocking
2. Try running as Administrator
3. Check if required ports (8000, 8501) are available
4. Verify Python version (3.8+ required)

## Success Indicators:
- Backend shows "Authentication routes enabled" 
- Frontend loads at http://localhost:8501
- Login with default accounts works
- API docs accessible at http://localhost:8000/docs
"""
    
    try:
        with open("FRIEND_SETUP_GUIDE.md", "w", encoding="utf-8") as f:
            f.write(guide)
        print("Setup guide created: FRIEND_SETUP_GUIDE.md")
    except UnicodeEncodeError:
        # Fallback with ASCII encoding if UTF-8 fails
        guide_ascii = guide.encode('ascii', 'ignore').decode('ascii')
        with open("FRIEND_SETUP_GUIDE.md", "w", encoding="ascii") as f:
            f.write(guide_ascii)
        print("Setup guide created with ASCII encoding: FRIEND_SETUP_GUIDE.md")
    except Exception as e:
        print(f"Error creating setup guide: {e}")

def run_diagnostics() -> bool:
    """Run complete diagnostic suite"""
    print("HEALTHCARE APP - AUTHENTICATION DIAGNOSTICS")
    print("=" * 60)
    
    # Check backend
    backend_ok = check_backend_status()
    
    if not backend_ok:
        print("\nCRITICAL: Backend is not running!")
        print("   Your friends need to start the backend first:")
        print("   python start.py")
        return False
    
    # Check auth endpoints
    endpoints = check_auth_endpoints()
    if not endpoints:
        print("\nCRITICAL: No auth endpoints found!")
        return False
    
    # Check user store
    store_ok = check_user_store()
    if not store_ok:
        print("\nWARNING: User store issues detected")
    
    # Test default users
    working_users = test_default_users()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if backend_ok and endpoints and working_users:
        print("AUTHENTICATION IS WORKING CORRECTLY")
        print(f"[OK] {len(working_users)} default accounts are functional")
        print("[OK] Your friends should be able to login")
        print("\nLIKELY ISSUE: Your friends' backend is not running")
        print("   Tell them to run: python start.py")
    else:
        print("AUTHENTICATION ISSUES DETECTED")
        print("   Check the errors above and fix them")
    
    # Create guide
    create_startup_guide()
    
    print(f"\nShare FRIEND_SETUP_GUIDE.md with your friends")
    print("   This contains step-by-step instructions")
    
    return True

if __name__ == "__main__":
    run_diagnostics()