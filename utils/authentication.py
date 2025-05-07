import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import hashlib
import uuid
import re

# File path for user database
USER_DB_PATH = "data/users.json"

def initialize_authentication():
    """Initialize the authentication system and create directories if needed"""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create user database file if it doesn't exist
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "w") as f:
            json.dump({"users": []}, f)

def load_users():
    """Load users from the database file"""
    with open(USER_DB_PATH, "r") as f:
        return json.load(f)

def save_users(users_data):
    """Save users to the database file"""
    with open(USER_DB_PATH, "w") as f:
        json.dump(users_data, f, indent=4)

def hash_password(password):
    """Hash password with SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email):
    """Check if email is valid"""
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_pattern, email))

def register_user(full_name, email, password, company=None):
    """Register a new user"""
    # Load current users
    users_data = load_users()
    
    # Check if email already exists
    for user in users_data["users"]:
        if user["email"] == email:
            return False, "Email already registered. Please use a different email."
    
    # Create new user
    new_user = {
        "id": str(uuid.uuid4()),
        "full_name": full_name,
        "email": email,
        "password_hash": hash_password(password),
        "company": company if company else "",
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    
    # Add user to database
    users_data["users"].append(new_user)
    save_users(users_data)
    
    return True, "Registration successful. You can now log in."

def authenticate_user(email, password):
    """Authenticate a user with email and password"""
    # Load users
    users_data = load_users()
    
    # Find user by email
    for user in users_data["users"]:
        if user["email"] == email and user["password_hash"] == hash_password(password):
            # Update last login time
            user["last_login"] = datetime.now().isoformat()
            save_users(users_data)
            
            return True, user
    
    return False, None

def update_profile(user_id, full_name=None, company=None):
    """Update user profile information"""
    users_data = load_users()
    
    for user in users_data["users"]:
        if user["id"] == user_id:
            if full_name:
                user["full_name"] = full_name
            if company is not None:  # Allow empty string
                user["company"] = company
            
            save_users(users_data)
            return True, "Profile updated successfully."
    
    return False, "User not found."

def change_password(user_id, old_password, new_password):
    """Change user password"""
    users_data = load_users()
    
    for user in users_data["users"]:
        if user["id"] == user_id:
            # Verify old password
            if user["password_hash"] != hash_password(old_password):
                return False, "Current password is incorrect."
            
            # Update password
            user["password_hash"] = hash_password(new_password)
            save_users(users_data)
            return True, "Password changed successfully."
    
    return False, "User not found."

def is_authenticated():
    """Check if user is authenticated"""
    return "user" in st.session_state and st.session_state.user is not None

def get_current_user():
    """Get current authenticated user"""
    if is_authenticated():
        return st.session_state.user
    return None

def login_user(user):
    """Log in the user by setting session state"""
    st.session_state.user = user

def logout_user():
    """Log out the user by clearing session state"""
    st.session_state.user = None
    # Also clear any other session data
    for key in ["data", "preprocessed_data", "transaction_data", "product_list",
                "association_rules", "forecasting_model", "predictions", "frequent_itemsets"]:
        if key in st.session_state:
            st.session_state[key] = None