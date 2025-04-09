#!/usr/bin/env python
# coding: utf-8

# # API for BodyTrust AI

# ### Save your trained model

# In[3]:


from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
print("✅ Secret key loaded:", SECRET_KEY)


# In[2]:


import joblib
joblib.dump(model, 'bodytrust_model.pkl')


# In[1]:


from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from pydantic import BaseModel
import joblib
import numpy as np

# === Constants for JWT ===
SECRET_KEY = "Ryomen_Sukuna"  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# === Dummy user database ===
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$7Ykd4o6biYX3w1dZ6YZrSOb.Tq8a2asVCgBTq2RKDrdihdKUT.BGa",  # password: admin123
    }
}

# === JWT / Auth Setup ===
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username not in fake_users_db:
            raise HTTPException(status_code=401, detail="User not found")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# === Pydantic Models ===
class Token(BaseModel):
    access_token: str
    token_type: str

class UserMetrics(BaseModel):
    Calories_y: float
    TotalMinutesAsleep: float
    BMI: float
    TotalSteps: float
    VeryActiveMinutes: float
    FairlyActiveMinutes: float
    LightlyActiveMinutes: float
    SedentaryMinutes: float

# === Load model & init app ===
model = joblib.load('bodytrust_model.pkl')
app = FastAPI(title="BodyTrust AI API with JWT Auth")

# === Routes ===

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token(data={"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predict_score")
async def predict_score(data: UserMetrics, username: str = Depends(get_current_user)):
    features = np.array([[ 
        data.Calories_y,
        data.TotalMinutesAsleep,
        data.BMI,
        data.TotalSteps,
        data.VeryActiveMinutes,
        data.FairlyActiveMinutes,
        data.LightlyActiveMinutes,
        data.SedentaryMinutes
    ]])
    predicted = model.predict(features)[0]
    return {
        "Predicted BodyTrust Score": round(predicted, 2),
        "Requested by": username
    }


# In[ ]:




