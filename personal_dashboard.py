#!/usr/bin/env python
# coding: utf-8

# ## Setup Dash App with dcc.Store and container

# In[2]:


import dash
from dash import dcc, html, Input, Output
import requests
import jwt

# Replace with your real secret key or load it from .env
SECRET_KEY = "your_super_secret_key"

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id="jwt-token", storage_type="session"),
    html.H1("🧠 BodyTrust AI Dashboard"),
    html.Div(id='dashboard-container')
])


# ## Callback to load personalized data if JWT exists

# In[3]:


@app.callback(
    Output('dashboard-container', 'children'),
    Input('jwt-token', 'data')
)
def display_dashboard(token):
    if not token:
        return html.Div([
            html.H4("🔒 Please log in to view your dashboard.")
        ])

    try:
        # Decode token to get username (optional but nice)
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = decoded.get("sub", "User")

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("http://127.0.0.1:8000/user-data", headers=headers)

        if response.status_code != 200:
            return html.Div(["❌ Failed to fetch user data."])

        data = response.json()

        return html.Div([
            html.H3(f"🎉 Welcome back, {username}!"),
            html.P(f"✅ Your BodyTrust Score: {data.get('bodytrust_score', 'N/A')}"),
            html.P(f"🚶‍♂️ Steps Today: {data.get('steps', 'N/A')}"),
            html.P(f"🔥 Calories Burned: {data.get('calories', 'N/A')}"),
            # You can even add Plotly charts here!
        ])

    except Exception as e:
        return html.Div([f"❌ Error: {str(e)}"])


# ## Run the app

# In[4]:


if __name__ == '__main__':
    app.run(debug=True, port=8052)


# In[ ]:




