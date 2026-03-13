from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from simulation.config import APP_ENV

app = FastAPI(
    title="HN26 AI Sandbox",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulateRequest(BaseModel):
    feature_description: str
    use_cache: Optional[bool] = False

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "env": APP_ENV,
        "version": "0.1.0",
    }

@app.post("/simulate")
def simulate(req: SimulateRequest):
    return {
        "status": "stub",
        "received": req.feature_description
    }