"""
Authentication components for the Streamlit frontend
"""
import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time


class AuthManager:
    """Handles authentication state and API calls"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.auth_endpoint = f"{backend_url}/api/auth"
    
    def login(self, email: str, password: str) -> tuple[bool, str, dict]:
        """
        Attempt to log in user
        Returns: (success, message, user_data)
        """
        try:
            response = requests.post(
                f"{self.auth_endpoint}/login",
                json={"email": email, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, "Login successful!", data
            elif response.status_code == 401:
                return False, "Invalid email or password", {}
            elif response.status_code == 423:
                return False, "Account is temporarily locked due to too many failed attempts", {}
            elif response.status_code == 403:
                return False, "Account is inactive. Please contact administrator.", {}
            else:
                error_detail = response.json().get("detail", "Login failed")
                return False, error_detail, {}
                
        except requests.RequestException as e:
            return False, f"Connection error: {str(e)}", {}
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", {}
    
    def logout(self, token: str) -> bool:
        """Log out user"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.post(
                f"{self.auth_endpoint}/logout",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return True  # Always allow logout on frontend
    
    def verify_token(self, token: str) -> tuple[bool, dict]:
        """Verify if token is still valid"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                f"{self.auth_endpoint}/verify-token",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {}
        except:
            return False, {}
    
    def get_user_info(self, token: str) -> Optional[dict]:
        """Get current user information"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                f"{self.auth_endpoint}/me",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except:
            return None
    
    def get_default_credentials(self) -> dict:
        """Get default test credentials"""
        try:
            response = requests.get(
                f"{self.auth_endpoint}/default-users",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except:
            return {}
    
    def signup(self, email: str, password: str, full_name: str, role: str = "patient") -> tuple[bool, str, dict]:
        """
        Attempt to sign up a new user
        Returns: (success, message, user_data)
        """
        try:
            # Use the public signup endpoint
            response = requests.post(
                f"{self.auth_endpoint}/signup",
                json={
                    "email": email, 
                    "password": password, 
                    "full_name": full_name, 
                    "role": role
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, "Account created successfully! Please log in.", data
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Signup failed")
                return False, error_detail, {}
            else:
                error_detail = response.json().get("detail", "Signup failed")
                return False, error_detail, {}
                
        except requests.RequestException as e:
            return False, f"Connection error: {str(e)}", {}
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", {}


def init_session_state():
    """Initialize authentication-related session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()


def check_session_validity():
    """Check if current session is still valid"""
    if not st.session_state.authenticated:
        return False
    
    # Check token expiry (basic check based on login time)
    if st.session_state.login_time:
        login_time = datetime.fromisoformat(st.session_state.login_time)
        if datetime.now() - login_time > timedelta(hours=1):
            logout_user()
            return False
    
    # Verify token with backend (optional, can be expensive)
    if st.session_state.access_token:
        valid, _ = st.session_state.auth_manager.verify_token(st.session_state.access_token)
        if not valid:
            logout_user()
            return False
    
    return True


def logout_user():
    """Log out the current user"""
    if st.session_state.access_token:
        st.session_state.auth_manager.logout(st.session_state.access_token)
    
    # Clear session state
    st.session_state.authenticated = False
    st.session_state.user_data = {}
    st.session_state.access_token = None
    st.session_state.login_time = None
    
    st.success("Logged out successfully!")
    st.rerun()


def show_development_credentials():
    """Show test credentials for development purposes only"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Development Mode")
    
    # Get default credentials
    default_creds = st.session_state.auth_manager.get_default_credentials()
    
    if default_creds:
        with st.sidebar.expander("🔑 Test Accounts", expanded=False):
            st.markdown("**Admin:**")
            st.code(f"{default_creds.get('admin', {}).get('email', 'N/A')}")
            st.code(f"{default_creds.get('admin', {}).get('password', 'N/A')}")
            
            st.markdown("**Doctor:**")
            st.code(f"{default_creds.get('doctor', {}).get('email', 'N/A')}")
            st.code(f"{default_creds.get('doctor', {}).get('password', 'N/A')}")
            
            st.markdown("**Patient:**")
            st.code(f"{default_creds.get('patient', {}).get('email', 'N/A')}")
            st.code(f"{default_creds.get('patient', {}).get('password', 'N/A')}")
            
            st.caption("⚠️ For development use only")


def render_login_page():
    """Render a clean and professional login page"""
    # Add custom CSS for login page styling
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-title {
        font-size: 2rem;
        color: #1a365d;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .form-container {
        margin-top: 1.5rem;
    }
    .tab-container {
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Header section
        st.markdown("""
        <div class="login-header">
            <h1 class="login-title">🏥 Healthcare System</h1>
            <p class="login-subtitle">Access your healthcare account</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tab selection for Login/Signup
        tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Sign Up"])
        
        with tab1:
            # Login form
            with st.form("login_form", clear_on_submit=False):
                st.markdown('<div class="form-container">', unsafe_allow_html=True)
                
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email address",
                    help="Use your registered healthcare system email",
                    label_visibility="visible"
                )
                
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    help="Your secure password",
                    label_visibility="visible"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Submit button with custom styling
                submitted = st.form_submit_button(
                    "Sign In",
                    type="primary",
                    use_container_width=True
                )
                
                if submitted:
                    if not email or not password:
                        st.error("⚠️ Please enter both email and password")
                    else:
                        with st.spinner("🔐 Authenticating..."):
                            success, message, user_data = st.session_state.auth_manager.login(email, password)
                        
                        if success:
                            # Store authentication data
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data.get('user', {})
                            st.session_state.access_token = user_data.get('token', {}).get('access_token')
                            st.session_state.login_time = datetime.now().isoformat()
                            
                            st.success(f"✅ Welcome, {st.session_state.user_data.get('full_name', 'User')}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
        
        with tab2:
            # Signup form
            with st.form("signup_form", clear_on_submit=False):
                st.markdown('<div class="form-container">', unsafe_allow_html=True)
                
                full_name = st.text_input(
                    "Full Name",
                    placeholder="Enter your full name",
                    help="Your complete name as it should appear in the system",
                    label_visibility="visible"
                )
                
                signup_email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email address",
                    help="This will be your login email",
                    label_visibility="visible"
                )
                
                signup_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Create a secure password",
                    help="Must be at least 8 characters with uppercase, lowercase, and numbers",
                    label_visibility="visible"
                )
                
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Confirm your password",
                    help="Re-enter your password to confirm",
                    label_visibility="visible"
                )
                
                role = st.selectbox(
                    "Account Type",
                    ["patient", "doctor", "nurse"],
                    help="Select your role in the healthcare system",
                    label_visibility="visible"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Submit button with custom styling
                signup_submitted = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True
                )
                
                if signup_submitted:
                    if not all([full_name, signup_email, signup_password, confirm_password]):
                        st.error("⚠️ Please fill in all fields")
                    elif signup_password != confirm_password:
                        st.error("⚠️ Passwords do not match")
                    elif len(signup_password) < 8:
                        st.error("⚠️ Password must be at least 8 characters long")
                    else:
                        with st.spinner("📝 Creating your account..."):
                            success, message, user_data = st.session_state.auth_manager.signup(
                                signup_email, signup_password, full_name, role
                            )
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.info("Please use the 'Sign In' tab to log in with your new account.")
                        else:
                            st.error(f"❌ {message}")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close login-container
    
    # System information in an expandable section below
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ About the Healthcare System", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Purpose**
            - AI-powered health risk assessment
            - Cardiovascular disease prediction
            - Personalized health recommendations
            
            **👥 User Roles**
            - **Admin**: System management
            - **Doctor**: Patient care & analytics
            - **Patient**: Personal health insights
            """)
        
        with col2:
            st.markdown("""
            **🔒 Security Features**
            - JWT token authentication
            - Role-based access control
            - Secure password encryption
            - Session management
            
            **📊 Key Features**
            - EHR data processing
            - Interactive dashboards
            - Real-time predictions
            - Health data visualization
            """)


def render_user_info():
    """Render user information in sidebar"""
    if st.session_state.authenticated and st.session_state.user_data:
        user = st.session_state.user_data
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Info")
        
        # User details
        st.sidebar.markdown(f"**Name:** {user.get('full_name', 'N/A')}")
        st.sidebar.markdown(f"**Email:** {user.get('email', 'N/A')}")
        st.sidebar.markdown(f"**Role:** {user.get('role', 'N/A').title()}")
        
        # Role badge
        role = user.get('role', '').lower()
        if role == 'admin':
            st.sidebar.markdown("🛡️ **Administrator**")
        elif role == 'doctor':
            st.sidebar.markdown("👨‍⚕️ **Healthcare Provider**")
        elif role == 'patient':
            st.sidebar.markdown("👤 **Patient**")
        
        # Login time
        if st.session_state.login_time:
            login_time = datetime.fromisoformat(st.session_state.login_time)
            st.sidebar.markdown(f"**Logged in:** {login_time.strftime('%H:%M:%S')}")
        
        st.sidebar.markdown("---")
        
        # Logout button
        if st.sidebar.button("🚪 Logout", type="primary", use_container_width=True):
            logout_user()


def require_auth(required_roles: Optional[list[str]] = None):
    """
    Decorator function to require authentication
    
    Args:
        required_roles: List of roles that can access the page (None = any authenticated user)
    """
    if not st.session_state.authenticated or not check_session_validity():
        render_login_page()
        st.stop()
    
    if required_roles:
        user_role = st.session_state.user_data.get('role', '').lower()
        if user_role not in [role.lower() for role in required_roles]:
            st.error(f"Access denied. Required role: {', '.join(required_roles)}")
            st.stop()


def get_auth_headers():
    """Get authentication headers for API requests"""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def is_admin():
    """Check if current user is admin"""
    return st.session_state.user_data.get('role', '').lower() == 'admin'


def is_doctor():
    """Check if current user is doctor or admin"""
    role = st.session_state.user_data.get('role', '').lower()
    return role in ['doctor', 'admin']


def is_patient():
    """Check if current user is patient"""
    return st.session_state.user_data.get('role', '').lower() == 'patient'