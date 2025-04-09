#!/usr/bin/env python
# coding: utf-8

# In[6]:


from dotenv import load_dotenv
import os

load_dotenv()

print("✅ DATABASE_URL loaded:", os.getenv("DATABASE_URL"))  # Should print the actual URL string


# In[1]:


from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# ✅ Correct way to get DATABASE_URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")  

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the User table
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String, default='user')


# In[ ]:




