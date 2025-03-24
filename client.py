from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os



jpl_bot = RemoteRunnable("http://localhost:8080/chat/")
question = 'Tell me about you self?'

print(jpl_bot.invoke(question).content) 