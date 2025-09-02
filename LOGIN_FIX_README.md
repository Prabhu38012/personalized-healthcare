# 🚨 LOGIN ISSUE FIX - FOR FRIENDS

If you're getting "not found" or login errors after pulling this project, here's the **quick fix**:

## 🔧 INSTANT FIX (Recommended)

### Windows Users:
```bash
# Just double-click this file:
fix_login.bat

# OR run in command prompt:
fix_login.bat
```

### Mac/Linux Users:
```bash
python fix_login_for_friends.py
```

## 🏃‍♂️ MANUAL FIX (If automatic doesn't work)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start Both Services

**Terminal 1 (Backend):**
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**  
```bash
python -m streamlit run frontend/app.py --server.port 8501
```

### Step 3: Login
1. Go to: **http://localhost:8501**
2. Use these accounts:

| Role | Email | Password |
|------|-------|----------|
| Admin | `admin@healthcare.com` | `Admin123!` |
| Doctor | `doctor@healthcare.com` | `Doctor123!` |
| Patient | `patient@healthcare.com` | `Patient123!` |

## 🔍 WHAT WAS THE PROBLEM?

The "not found" error happens because:
1. **Backend not running** - The login system needs the backend API
2. **Wrong order** - You started frontend before backend  
3. **Port conflicts** - Something else was using port 8000

## ✅ SUCCESS INDICATORS

You'll know it's working when:
- ✅ Backend shows: "Authentication routes enabled"
- ✅ Frontend loads at http://localhost:8501
- ✅ Login page appears (not error page)
- ✅ You can login with the test accounts above

## 🆘 STILL HAVING ISSUES?

### Quick Health Check:
```bash
# Test if backend is responding:
curl http://localhost:8000/api/health

# Test auth service:
curl http://localhost:8000/api/auth/test-connection
```

### Common Solutions:
1. **Run as Administrator** (Windows)
2. **Check firewall** - Allow Python through Windows Defender
3. **Kill processes** using ports 8000/8501
4. **Python version** - Must be 3.8 or higher

### Get Help:
- Check the console output for specific error messages
- Look for "Authentication routes enabled" in backend logs
- Make sure you're in the `personalized-healthcare` directory

---
**💡 TIP**: The automatic fix script (`fix_login_for_friends.py`) will diagnose and fix most issues automatically!