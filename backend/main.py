from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="AI Sandbox for Product Impact Analysis", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "env": os.getenv("ENV", "unknown")}

@app.get("/population/segments")
def get_segments():
    return {"message": "Population engine not yet initialized"}

@app.post("/simulate")
def simulate(payload: dict):
    return {"message": "Simulation engine not yet initialized", "received": payload}