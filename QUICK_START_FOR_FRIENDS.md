# Quick Start Guide - After Pulling Latest Updates

## What Your Friends Need to Do After Pulling Your Changes

### Step 1: Pull the Latest Code
```bash
git pull origin main
```

### Step 2: Set Up Environment (First Time Only)
If they haven't set up the environment before:

**Windows:**
```bash
# Navigate to project directory
cd personalized-healthcare

# Run the setup script
setup.bat

# OR manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Start the Application
Your friends have several options:

#### Option A: Use the Startup Script (Recommended)
```bash
python start.py
```
Then select option 3 (Both backend and frontend)

#### Option B: Use the Fixed Debug Script
```bash
python debug_auth.py
```
This will run diagnostics and show them exactly what to do.

#### Option C: Use the Automated Fix Tool
```bash
python fix_login_for_friends.py
```
This will automatically set everything up and start both services.

#### Option D: Manual Startup
```bash
# Terminal 1 - Start Backend:
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Start Frontend:
python -m streamlit run frontend/app.py --server.port 8501
```

### Step 4: Access the Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Step 5: Login with Test Accounts
Use these working accounts:

**Admin Account:**
- Email: `admin@healthcare.com`
- Password: `Admin123!`

**Doctor Account:**
- Email: `doctor@healthcare.com`
- Password: `Doctor123!`

**Patient Account:**
- Email: `patient@healthcare.com`
- Password: `Patient123!`

## Quick Troubleshooting

### If They Get "Not Found" Errors:
1. Make sure the backend is running (step 3)
2. Check they're using the correct URL: http://localhost:8501
3. Verify no antivirus is blocking Python

### If They Have Issues:
1. Run the debug script: `python debug_auth.py`
2. Run the fix script: `python fix_login_for_friends.py`
3. Check the generated `FRIEND_SETUP_GUIDE.md`

### Quick Test:
```bash
# Test if everything is working:
python simple_auth_test.py
```

## What's Fixed in This Update:
- ✅ Unicode encoding errors resolved
- ✅ Authentication system verified working
- ✅ Improved Windows compatibility
- ✅ Better error messages and diagnostics
- ✅ Automated setup tools provided

## Success Indicators:
- Backend shows "Authentication routes enabled"
- Frontend loads without errors
- Login works with the test accounts above
- No Unicode/encoding errors in terminal

---

**Note:** If your friends are still having issues, they should run `python debug_auth.py` which will diagnose the problem and provide specific solutions.