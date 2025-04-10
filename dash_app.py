#!/usr/bin/env python
# coding: utf-8

# # Plotly Dash dashboard using your bodytrust_ai_final.csv. It has:
# ✅ Dropdown filter for HealthTier
# 
# ✅ Scatter plot with predicted score as marker size
# 
# ✅ Bar chart comparing actual vs predicted scores
# 
# ✅ Recommendation table sorted by ScoreDelta

# In[5]:


import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

# Load data
cluster_df = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/Python/bodytrust_ai_final.csv")

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "BodyTrust AI Dashboard"

# Layout
app.layout = html.Div([
    html.H1("💪 BodyTrust AI Dashboard", style={"textAlign": "center"}),
html.Div([
    html.Img(
        src='/assets/bodytrust_logo.png',
        style={
            'width': '200px',
            'display': 'block',
            'margin': 'auto',
            'padding-top': '10px'
        }
    )
]),

    html.Div([
        html.Label("Filter by Health Tier:"),
        dcc.Dropdown(
            id='health-tier-filter',
            options=[
                {'label': tier, 'value': tier} for tier in cluster_df['HealthTier'].unique()
            ],
            value=None,
            placeholder="Select Health Tier",
            multi=False
        )
    ], style={"width": "30%", "margin": "auto"}),

    html.Br(),

    dcc.Store(id="jwt-token", storage_type="session"),

    html.Div(id="admin-panel", children=[
        html.H3("👑 Admin Panel"),
        html.P("This section is only visible to admins."),

        html.H4("All Users"),
dash_table.DataTable(
    id="user-admin-table",
    columns=[
        {"name": "Username", "id": "username"},
        {"name": "Email", "id": "email"},
        {"name": "Role", "id": "role"},
    ],
    style_table={'overflowX': 'auto'},
    page_size=5
),

html.Br(),
html.Label("Promote user to admin:"),
dcc.Dropdown(id="promote-user-dropdown", placeholder="Select username"),
html.Button("Promote", id="promote-btn", n_clicks=0),
html.Div(id="promote-response")
        
        # Add admin tools here
    ], style={"display": "none"}),

    html.Br(),

    dcc.Graph(id='trust-vs-calories'),

    html.Br(),

    dcc.Graph(id='avg-score-bar'),

    html.Br(),

    html.H3("📋 Users & Recommendations", style={"textAlign": "center"}),
    dash_table.DataTable(
        id='user-table',
        columns=[
            {'name': col, 'id': col} for col in [
                'Id', 'HealthTier', 'BodyTrustScore', 'PredictedScore', 'ScoreDelta', 'Recommendation'
            ]
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '5px',
            'fontFamily': 'Arial',
            'fontSize': '14px'
        },
        page_size=10
    )
])
# ✅ Admin role check callback (place it here)
@app.callback(
    Output("admin-panel", "style"),
    Input("jwt-token", "data")
)
def check_admin_role(token):
    if not token:
        return {"display": "none"}

    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        role = decoded.get("role", "user")

        if role == "admin":
            return {"display": "block"}
        else:
            return {"display": "none"}

    except Exception as e:
        return {"display": "none"}
# Callbacks
@app.callback(
    [
        Output('trust-vs-calories', 'figure'),
        Output('avg-score-bar', 'figure'),
        Output('user-table', 'data')
    ],
    [Input('health-tier-filter', 'value')]
)
def update_dashboard(selected_tier):
    if selected_tier:
        filtered_df = cluster_df[cluster_df['HealthTier'] == selected_tier]
    else:
        filtered_df = cluster_df

    # Scatter plot
    scatter_fig = px.scatter(
        filtered_df,
        x='BodyTrustScore',
        y='Calories_y',
        color='HealthTier',
        hover_data=['Id', 'BMI', 'TotalMinutesAsleep'],
        title='BodyTrust Score vs Calories Burned',
        size='PredictedScore',
        labels={'Calories_y': 'Calories Burned'}
    )

    # Bar chart
    avg_df = cluster_df.groupby('HealthTier')[['BodyTrustScore', 'PredictedScore']].mean().reset_index()
    bar_fig = px.bar(
        avg_df,
        x='HealthTier',
        y=['BodyTrustScore', 'PredictedScore'],
        barmode='group',
        title='Average Actual vs Predicted Score by Health Tier',
        labels={'value': 'Score', 'variable': 'Score Type'}
    )

    # Table
    table_data = filtered_df.sort_values(by='ScoreDelta', ascending=True).to_dict('records')

    return scatter_fig, bar_fig, table_data

# Run the server
if __name__ == '__main__':
    app.run(debug=True, port=8053)


# In[ ]:




