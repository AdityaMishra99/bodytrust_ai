#!/usr/bin/env python
# coding: utf-8

# In[4]:


from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from auth import hash_password, verify_password, create_access_token, decode_token
from database import SessionLocal, engine, User  # from your db setup
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dependencies import get_current_user, admin_required
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == user.username) | (User.email == user.email)).first():
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User registered successfully!"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token_data = {
        "sub": user.username,
        "role": user.role  # 👈 Include role in JWT
    }

    access_token = create_access_token(token_data)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users")
def get_all_users(user: User = Depends(admin_required), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"username": u.username, "email": u.email, "role": u.role} for u in users]

@app.post("/promote")
def promote_user(username: str, user: User = Depends(admin_required), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.username == username).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    target.role = "admin"
    db.commit()
    return {"msg": f"{username} promoted to admin."}

@app.get("/me")
def get_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = decode_token(token)
        username = payload.get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": user.username, "email": user.email}


# In[ ]:




