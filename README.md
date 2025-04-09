<p align="center">
  <img src="assets/bodytrust_logo.png" width="300"/>
</p>


# 💪 BodyTrust AI

**Earn trust in health like you do in finance – with data.**

BodyTrust AI is a real-time health analytics platform built using Python, Dash, and FastAPI. It uses wearable data (like steps, calories, sleep, BMI) to generate a **BodyTrust Score**, classify health tiers, and predict future performance using machine learning.

---

## ⚙️ Key Features

- 🔐 Secure Login & JWT Authentication
- 📊 BodyTrust Score based on real health metrics
- 🤖 ML model predicts future scores (RandomForest)
- 👑 Admin panel: user management + access control
- 🧠 Clustering with KMeans for tier classification
- 💻 Dash frontend + FastAPI backend + PostgreSQL DB

---

## 🛠️ Tech Stack

- Python, Plotly Dash
- FastAPI (JWT-secured)
- PostgreSQL + SQLAlchemy
- Machine Learning (RandomForestRegressor)
- KMeans Clustering
- Role-based Dashboards

---

## 📂 Run Locally

```bash
git clone https://github.com/AdityaMishra99/bodytrust_ai.git
cd bodytrust_ai

# Setup virtualenv + install
pip install -r requirements.txt

# Start FastAPI backend
uvicorn api_server:app --reload

# Start Dash frontend
python dash_app.py
