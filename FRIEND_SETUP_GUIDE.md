
# 🏥 Healthcare App - Quick Start Guide for Friends

## Step 1: Setup Environment
```bash
# Navigate to project directory
cd personalized-healthcare

# Run setup script (Windows)
.\setup.bat

# OR manually create virtual environment
python -m venv venv
venv\Scripts\activate
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
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@healthcare.com\",\"password\":\"Admin123!\"}"
```

### If Still Having Issues:
1. Check Windows Defender/Antivirus isn't blocking
2. Try running as Administrator
3. Check if required ports (8000, 8501) are available
4. Verify Python version (3.8+ required)

## Success Indicators:
- ✅ Backend shows "Authentication routes enabled" 
- ✅ Frontend loads at http://localhost:8501
- ✅ Login with default accounts works
- ✅ API docs accessible at http://localhost:8000/docs
