#!/usr/bin/env python
# coding: utf-8

# #  Add a dcc.Store(id="jwt-token", storage_type="session") to your layout

# In[1]:


import jwt
import os
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

SECRET_KEY = os.getenv("SECRET_KEY") or "your_dev_key"

# 🔥 Define your Dash app here
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # if you're deploying later (e.g., on Heroku)

app.layout = html.Div([
    dcc.Input(id='username-input', placeholder='Enter username', type='text'),
    dcc.Input(id='password-input', placeholder='Enter password', type='password'),
    html.Button('Login', id='login-button'),
    html.Div(id='login-status'),
    dcc.Store(id='jwt-token', storage_type='session'),  # ✅ for saving the token
    html.Div(id='user-dashboard')  # Show user data after login
])


# # Update your login callback to store the JWT token

# In[4]:


@app.callback(
    Output("login-status", "children"),
    Output("jwt-token", "data"),  # ✅ Save token to dcc.Store
    Input("login-button", "n_clicks"),
    State("username-input", "value"),
    State("password-input", "value"),
    prevent_initial_call=True
)
def login_user(n_clicks, username, password):
    if username and password:
        res = requests.post(
            "http://localhost:8000/login",  # 🔄 Your FastAPI login route
            json={"username": username, "password": password}
        )
        if res.status_code == 200:
            token = res.json()["access_token"]
            return f"✅ Welcome, {username}!", token
        else:
            return f"❌ Login failed: {res.text}", None
    return "⚠️ Please enter username and password", None


# #  Use the token in future callbacks for authenticated API calls

# In[5]:


@app.callback(
    Output("user-dashboard", "children"),
    Input("jwt-token", "data"),
)
def show_dashboard(token):
    if token:
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get("http://localhost:8000/protected-route", headers=headers)
        if res.status_code == 200:
            return f"👋 Dashboard Loaded! {res.json()}"
        else:
            return "⚠️ Unauthorized or expired token"
    return "🔐 Please log in to access the dashboard."

