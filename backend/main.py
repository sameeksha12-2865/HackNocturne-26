"""
FastAPI Backend — Layer 5
API layer with built-in Requestly integration hooks and API testing support.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.population import generate_population, validate_population, save_population
from simulation.feature_interpreter import interpret_feature, FeatureSignal
from simulation.hifi_simulator import (
    stratified_sample, run_simulation_sync, build_training_dataset
)
from models.approximator import (
    predict_rule_based_all, compute_segment_aggregates, compute_gini, train_and_predict
)


# ── App Setup ──────────────────────────────────────────────────────

app = FastAPI(
    title="AI Sandbox — Product Impact Analysis",
    description="Simulate product decisions before deployment. Built for Hack-Nocturne '26.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id", "X-Pipeline-Stage", "X-Processing-Time-Ms",
                     "X-Simulation-Mode", "X-Population-Size", "X-Requestly-Mock-Eligible"],
)


# ── State ──────────────────────────────────────────────────────────

population: list[dict] = []
population_stats: dict = {}
latest_results: Optional[dict] = None
simulation_history: list[dict] = []  # Track all simulations for comparison
api_call_log: list[dict] = []  # Log all API interactions for Requestly debugging


# ── Requestly Integration Middleware ───────────────────────────────

@app.middleware("http")
async def requestly_middleware(request: Request, call_next):
    """
    Middleware that powers Requestly integration:
    - Assigns unique request IDs for tracing
    - Tracks pipeline stages in response headers
    - Logs all requests for Requestly API collection debugging
    - Supports mock mode via X-Requestly-Mock header
    - Adds timing headers for performance monitoring in Requestly
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Check if Requestly is requesting a mock response
    mock_mode = request.headers.get("X-Requestly-Mock", "false").lower() == "true"

    # Store request context
    request.state.request_id = request_id
    request.state.mock_mode = mock_mode
    request.state.start_time = start_time

    response = await call_next(request)

    # Add Requestly-friendly headers
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
    response.headers["X-Requestly-Mock-Eligible"] = "true"

    # Log for API debugging (accessible via /api/request-log)
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": str(request.url.path),
        "mock_mode": mock_mode,
        "processing_time_ms": elapsed_ms,
        "status_code": response.status_code,
    }
    api_call_log.append(log_entry)
    if len(api_call_log) > 100:
        api_call_log.pop(0)

    return response


# ── Request/Response Models ────────────────────────────────────────

class SimulateRequest(BaseModel):
    description: str
    use_llm: bool = False
    population_size: int = 2000


class SimulateResponse(BaseModel):
    request_id: str
    feature_signal: dict
    segments: dict
    gini_satisfaction: float
    gini_engagement: float
    population_size: int
    simulation_mode: str
    timestamp: str
    predictions_summary: dict
    weekly_projections: list[dict]


# ── Startup ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global population, population_stats
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'population.json')
    stats_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'population_stats.json')

    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            population = json.load(f)
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                population_stats = json.load(f)
        else:
            population_stats = validate_population(population)
        print(f"  Loaded {len(population)} users from existing data")
    else:
        print("  Generating fresh population...")
        population = generate_population(2000)
        population_stats = validate_population(population)
        save_population(population, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
        with open(stats_path, 'w') as f:
            json.dump(population_stats, f, indent=2)
        print(f"  Generated {len(population)} users")


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used by Requestly to verify backend is running."""
    return {
        "status": "healthy",
        "population_loaded": len(population),
        "simulations_run": len(simulation_history),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/population/segments")
async def get_segments():
    """Return population segment distribution stats."""
    return {
        "total_users": len(population),
        "segments": population_stats.get("segments", {}),
    }


@app.post("/simulate")
async def simulate(req: SimulateRequest, request: Request):
    """
    Main simulation endpoint.
    Accepts a feature description, runs the full 5-layer pipeline.
    
    Requestly Integration Points:
    - Set X-Requestly-Mock: true header to get pre-computed mock results
    - X-Pipeline-Stage header tracks which stage the request is in
    - X-Request-Id for tracing through the pipeline
    """
    global latest_results
    request_id = request.state.request_id
    mock_mode = request.state.mock_mode

    # ── Mock mode for Requestly ──
    if mock_mode:
        mock_result = _get_mock_result(req.description)
        mock_result["request_id"] = request_id
        mock_result["simulation_mode"] = "mock (Requestly)"
        latest_results = mock_result
        return mock_result

    try:
        # Stage 1: Feature interpretation
        signal = interpret_feature(req.description, os.getenv('GEMINI_API_KEY'))
        signal_dict = signal.model_dump()
        # Convert enum values to strings for JSON serialization
        signal_dict['change_type'] = signal_dict['change_type'].value if hasattr(signal_dict['change_type'], 'value') else signal_dict['change_type']
        signal_dict['direction_by_segment'] = {
            k: (v.value if hasattr(v, 'value') else v)
            for k, v in signal_dict['direction_by_segment'].items()
        }

        # Stage 2: Simulation (stratified sample → rule-based or LLM)
        sampled = stratified_sample(population, per_segment=6)
        sim_results = run_simulation_sync(
            sampled, signal_dict, use_llm=req.use_llm,
            api_key=os.getenv('GEMINI_API_KEY')
        )

        # Stage 3: Scale to full population
        if len(sim_results) >= 30 and req.use_llm:
            training_data = build_training_dataset(population, sim_results, signal_dict)
            predictions = train_and_predict(training_data, population, signal_dict)
        else:
            predictions = predict_rule_based_all(population, signal_dict)

        # Stage 4: Analytics
        aggregates = compute_segment_aggregates(predictions)
        sat_deltas = [p['delta_satisfaction'] for p in predictions]
        eng_deltas = [p['delta_engagement'] for p in predictions]
        gini_sat = compute_gini(sat_deltas)
        gini_eng = compute_gini(eng_deltas)

        # Weekly projections (compounding over 4 weeks)
        weekly_projections = []
        for week in range(1, 5):
            week_data = {}
            for seg, agg in aggregates.items():
                week_data[seg] = {
                    "engagement_delta": round(agg["engagement"]["mean"] * week * 0.85, 2),
                    "churn_rate": round(agg["churn_rate"] * (1 + 0.15 * (week - 1)), 1),
                    "satisfaction_delta": round(agg["satisfaction"]["mean"] * week * 0.8, 2),
                }
            weekly_projections.append({"week": week, "segments": week_data})

        # Build response
        total_churners = sum(1 for p in predictions if p['will_churn'])
        result = {
            "request_id": request_id,
            "feature_signal": signal_dict,
            "segments": aggregates,
            "gini_satisfaction": gini_sat,
            "gini_engagement": gini_eng,
            "population_size": len(predictions),
            "simulation_mode": "llm" if req.use_llm else "rule-based",
            "timestamp": datetime.now().isoformat(),
            "predictions_summary": {
                "total_users": len(predictions),
                "total_churners": total_churners,
                "overall_churn_rate": round(total_churners / len(predictions) * 100, 1),
                "avg_engagement_delta": round(sum(p['delta_engagement'] for p in predictions) / len(predictions), 2),
                "avg_satisfaction_delta": round(sum(p['delta_satisfaction'] for p in predictions) / len(predictions), 2),
            },
            "weekly_projections": weekly_projections,
        }

        latest_results = result
        simulation_history.append({
            "description": req.description,
            "timestamp": result["timestamp"],
            "result": result,
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/latest")
async def get_latest_results():
    """Return latest simulation results."""
    if latest_results is None:
        raise HTTPException(status_code=404, detail="No simulation has been run yet")
    return latest_results


@app.get("/results/history")
async def get_history():
    """Return all simulation history for side-by-side comparison."""
    return {
        "count": len(simulation_history),
        "simulations": [
            {
                "description": s["description"],
                "timestamp": s["timestamp"],
                "gini_satisfaction": s["result"]["gini_satisfaction"],
                "overall_churn_rate": s["result"]["predictions_summary"]["overall_churn_rate"],
            }
            for s in simulation_history
        ]
    }


# ── Requestly-Specific Endpoints ──────────────────────────────────

@app.get("/api/request-log")
async def get_request_log():
    """
    Returns a log of all API requests for Requestly debugging and monitoring.
    Use this in Requestly to validate request/response patterns.
    """
    return {
        "total_requests": len(api_call_log),
        "requests": api_call_log[-20:],  # Last 20 entries
    }


@app.get("/api/requestly-collection")
async def get_requestly_collection():
    """
    Returns a Requestly-compatible API collection definition.
    Import this directly into Requestly to set up all your test requests.
    """
    base_url = "{{BASE_URL}}"
    return {
        "name": "HN26-Sandbox",
        "description": "AI Sandbox for Product Impact Analysis — API Collection",
        "variables": {
            "BASE_URL": "http://localhost:8000",
        },
        "requests": [
            {
                "name": "Health Check",
                "method": "GET",
                "url": f"{base_url}/health",
                "description": "Verify backend is running",
                "tests": [
                    "assert response.status === 200",
                    "assert response.json.status === 'healthy'",
                    "assert response.json.population_loaded > 0",
                ],
            },
            {
                "name": "Get Population Segments",
                "method": "GET",
                "url": f"{base_url}/population/segments",
                "description": "Get segment distribution statistics",
                "tests": [
                    "assert response.status === 200",
                    "assert Object.keys(response.json.segments).length === 5",
                    "assert response.json.total_users >= 2000",
                ],
            },
            {
                "name": "Simulate — Pricing Change",
                "method": "POST",
                "url": f"{base_url}/simulate",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "description": "Increase premium subscription price by 15% and remove free tier export feature",
                    "use_llm": False,
                }),
                "description": "Run a pricing change simulation",
                "tests": [
                    "assert response.status === 200",
                    "assert response.json.segments !== undefined",
                    "assert Object.keys(response.json.segments).length > 0",
                    "assert response.json.gini_satisfaction >= 0",
                    "assert response.json.gini_satisfaction <= 1",
                ],
            },
            {
                "name": "Simulate — New Feature",
                "method": "POST",
                "url": f"{base_url}/simulate",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "description": "Adding AI-powered search for all tiers",
                    "use_llm": False,
                }),
                "description": "Run a new feature simulation",
                "tests": [
                    "assert response.status === 200",
                    "assert response.json.simulation_mode !== undefined",
                ],
            },
            {
                "name": "Simulate — Mock Mode (Requestly)",
                "method": "POST",
                "url": f"{base_url}/simulate",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Requestly-Mock": "true",
                },
                "body": json.dumps({
                    "description": "Any description — mock mode returns cached results",
                    "use_llm": False,
                }),
                "description": "Test mock mode: returns pre-computed results without running pipeline",
                "tests": [
                    "assert response.status === 200",
                    "assert response.json.simulation_mode === 'mock (Requestly)'",
                ],
            },
            {
                "name": "Get Latest Results",
                "method": "GET",
                "url": f"{base_url}/results/latest",
                "description": "Get the most recent simulation results",
                "tests": [
                    "assert response.status === 200 || response.status === 404",
                ],
            },
            {
                "name": "Get Simulation History",
                "method": "GET",
                "url": f"{base_url}/results/history",
                "description": "Get all past simulations for comparison",
                "tests": [
                    "assert response.status === 200",
                    "assert Array.isArray(response.json.simulations)",
                ],
            },
            {
                "name": "API Request Log",
                "method": "GET",
                "url": f"{base_url}/api/request-log",
                "description": "View all API requests — useful for Requestly debugging",
                "tests": [
                    "assert response.status === 200",
                    "assert response.json.total_requests >= 0",
                ],
            },
        ],
    }


@app.get("/api/mock-data/{scenario}")
async def get_mock_data(scenario: str):
    """
    Serves pre-computed mock data for different scenarios.
    Requestly can redirect /simulate to this endpoint when LLM credits run low.
    
    Scenarios: 'pricing_increase', 'new_feature', 'feature_removal'
    """
    return _get_mock_result(scenario)


# ── Mock Data Generator ──────────────────────────────────────────

def _get_mock_result(description: str) -> dict:
    """Generate realistic mock results for demo resilience."""
    desc_lower = description.lower()

    if any(w in desc_lower for w in ['price', 'cost', 'increase', 'pricing_increase']):
        return {
            "request_id": "mock-001",
            "feature_signal": {
                "feature_name": "Price increase 15%",
                "change_type": "pricing",
                "affected_dimensions": ["churn_risk", "satisfaction", "spend"],
                "direction_by_segment": {
                    "power_user": "neutral",
                    "casual_browser": "negative",
                    "price_sensitive": "negative",
                    "early_adopter": "neutral",
                    "enterprise_user": "neutral",
                },
                "magnitude_estimate": 0.7,
                "confidence": 0.8,
            },
            "segments": {
                "power_user": {
                    "count": 400, "engagement": {"mean": -1.2, "std": 3.1, "p10": -5.0, "p90": 2.5},
                    "churn_risk": {"mean": 0.005, "std": 0.01, "p10": -0.005, "p90": 0.02},
                    "satisfaction": {"mean": -3.5, "std": 5.2, "p10": -10.0, "p90": 2.0},
                    "churn_count": 12, "churn_rate": 3.0,
                },
                "casual_browser": {
                    "count": 500, "engagement": {"mean": -8.5, "std": 6.2, "p10": -18.0, "p90": -1.0},
                    "churn_risk": {"mean": 0.045, "std": 0.02, "p10": 0.01, "p90": 0.08},
                    "satisfaction": {"mean": -15.2, "std": 7.5, "p10": -25.0, "p90": -5.0},
                    "churn_count": 85, "churn_rate": 17.0,
                },
                "price_sensitive": {
                    "count": 500, "engagement": {"mean": -16.3, "std": 8.0, "p10": -28.0, "p90": -5.0},
                    "churn_risk": {"mean": 0.12, "std": 0.05, "p10": 0.04, "p90": 0.20},
                    "satisfaction": {"mean": -22.8, "std": 9.3, "p10": -35.0, "p90": -10.0},
                    "churn_count": 185, "churn_rate": 37.0,
                },
                "early_adopter": {
                    "count": 300, "engagement": {"mean": 1.5, "std": 4.0, "p10": -3.5, "p90": 6.0},
                    "churn_risk": {"mean": -0.002, "std": 0.008, "p10": -0.01, "p90": 0.008},
                    "satisfaction": {"mean": -2.1, "std": 4.5, "p10": -8.0, "p90": 3.0},
                    "churn_count": 8, "churn_rate": 2.7,
                },
                "enterprise_user": {
                    "count": 300, "engagement": {"mean": -0.5, "std": 2.5, "p10": -3.5, "p90": 2.0},
                    "churn_risk": {"mean": 0.003, "std": 0.006, "p10": -0.003, "p90": 0.01},
                    "satisfaction": {"mean": -4.0, "std": 3.8, "p10": -9.0, "p90": 0.5},
                    "churn_count": 6, "churn_rate": 2.0,
                },
            },
            "gini_satisfaction": 0.41,
            "gini_engagement": 0.35,
            "population_size": 2000,
            "simulation_mode": "mock",
            "timestamp": datetime.now().isoformat(),
            "predictions_summary": {
                "total_users": 2000,
                "total_churners": 296,
                "overall_churn_rate": 14.8,
                "avg_engagement_delta": -6.7,
                "avg_satisfaction_delta": -11.2,
            },
            "weekly_projections": [
                {"week": 1, "segments": {
                    "power_user": {"engagement_delta": -1.0, "churn_rate": 3.0, "satisfaction_delta": -3.0},
                    "casual_browser": {"engagement_delta": -7.2, "churn_rate": 17.0, "satisfaction_delta": -12.2},
                    "price_sensitive": {"engagement_delta": -13.9, "churn_rate": 37.0, "satisfaction_delta": -18.2},
                    "early_adopter": {"engagement_delta": 1.3, "churn_rate": 2.7, "satisfaction_delta": -1.7},
                    "enterprise_user": {"engagement_delta": -0.4, "churn_rate": 2.0, "satisfaction_delta": -3.2},
                }},
                {"week": 2, "segments": {
                    "power_user": {"engagement_delta": -2.0, "churn_rate": 3.5, "satisfaction_delta": -5.6},
                    "casual_browser": {"engagement_delta": -14.5, "churn_rate": 19.6, "satisfaction_delta": -24.3},
                    "price_sensitive": {"engagement_delta": -27.7, "churn_rate": 42.6, "satisfaction_delta": -36.5},
                    "early_adopter": {"engagement_delta": 2.6, "churn_rate": 3.1, "satisfaction_delta": -3.4},
                    "enterprise_user": {"engagement_delta": -0.9, "churn_rate": 2.3, "satisfaction_delta": -6.4},
                }},
                {"week": 3, "segments": {
                    "power_user": {"engagement_delta": -3.1, "churn_rate": 3.9, "satisfaction_delta": -8.4},
                    "casual_browser": {"engagement_delta": -21.7, "churn_rate": 22.1, "satisfaction_delta": -36.5},
                    "price_sensitive": {"engagement_delta": -41.6, "churn_rate": 48.1, "satisfaction_delta": -50.0},
                    "early_adopter": {"engagement_delta": 3.8, "churn_rate": 3.5, "satisfaction_delta": -5.0},
                    "enterprise_user": {"engagement_delta": -1.3, "churn_rate": 2.6, "satisfaction_delta": -9.6},
                }},
                {"week": 4, "segments": {
                    "power_user": {"engagement_delta": -4.1, "churn_rate": 4.4, "satisfaction_delta": -11.2},
                    "casual_browser": {"engagement_delta": -28.9, "churn_rate": 24.7, "satisfaction_delta": -48.6},
                    "price_sensitive": {"engagement_delta": -50.0, "churn_rate": 53.7, "satisfaction_delta": -50.0},
                    "early_adopter": {"engagement_delta": 5.1, "churn_rate": 3.9, "satisfaction_delta": -6.7},
                    "enterprise_user": {"engagement_delta": -1.7, "churn_rate": 2.9, "satisfaction_delta": -12.8},
                }},
            ],
        }
    else:
        # Generic positive feature mock
        return {
            "request_id": "mock-002",
            "feature_signal": {
                "feature_name": description[:60],
                "change_type": "new_feature",
                "affected_dimensions": ["engagement", "satisfaction"],
                "direction_by_segment": {seg: "positive" for seg in
                    ["power_user", "casual_browser", "price_sensitive", "early_adopter", "enterprise_user"]},
                "magnitude_estimate": 0.5,
                "confidence": 0.7,
            },
            "segments": {
                "power_user": {
                    "count": 400, "engagement": {"mean": 8.5, "std": 4.0, "p10": 3.0, "p90": 14.0},
                    "churn_risk": {"mean": -0.015, "std": 0.008, "p10": -0.025, "p90": -0.005},
                    "satisfaction": {"mean": 10.2, "std": 5.0, "p10": 4.0, "p90": 16.0},
                    "churn_count": 4, "churn_rate": 1.0,
                },
                "casual_browser": {
                    "count": 500, "engagement": {"mean": 3.2, "std": 3.5, "p10": -1.0, "p90": 7.0},
                    "churn_risk": {"mean": -0.008, "std": 0.01, "p10": -0.02, "p90": 0.005},
                    "satisfaction": {"mean": 5.5, "std": 4.2, "p10": 0.5, "p90": 10.0},
                    "churn_count": 20, "churn_rate": 4.0,
                },
                "price_sensitive": {
                    "count": 500, "engagement": {"mean": 5.0, "std": 5.0, "p10": -2.0, "p90": 12.0},
                    "churn_risk": {"mean": -0.02, "std": 0.015, "p10": -0.04, "p90": 0.005},
                    "satisfaction": {"mean": 7.8, "std": 5.5, "p10": 1.0, "p90": 14.0},
                    "churn_count": 15, "churn_rate": 3.0,
                },
                "early_adopter": {
                    "count": 300, "engagement": {"mean": 12.0, "std": 5.0, "p10": 5.0, "p90": 18.0},
                    "churn_risk": {"mean": -0.02, "std": 0.008, "p10": -0.03, "p90": -0.01},
                    "satisfaction": {"mean": 14.5, "std": 4.5, "p10": 8.0, "p90": 20.0},
                    "churn_count": 2, "churn_rate": 0.7,
                },
                "enterprise_user": {
                    "count": 300, "engagement": {"mean": 4.5, "std": 3.0, "p10": 1.0, "p90": 8.0},
                    "churn_risk": {"mean": -0.01, "std": 0.005, "p10": -0.018, "p90": -0.002},
                    "satisfaction": {"mean": 6.0, "std": 3.5, "p10": 2.0, "p90": 10.0},
                    "churn_count": 3, "churn_rate": 1.0,
                },
            },
            "gini_satisfaction": 0.18,
            "gini_engagement": 0.22,
            "population_size": 2000,
            "simulation_mode": "mock",
            "timestamp": datetime.now().isoformat(),
            "predictions_summary": {
                "total_users": 2000,
                "total_churners": 44,
                "overall_churn_rate": 2.2,
                "avg_engagement_delta": 5.8,
                "avg_satisfaction_delta": 7.9,
            },
            "weekly_projections": [
                {"week": w, "segments": {
                    seg: {
                        "engagement_delta": round(5.8 * w * 0.85, 2),
                        "churn_rate": round(2.2 * (1 + 0.15 * (w - 1)), 1),
                        "satisfaction_delta": round(7.9 * w * 0.8, 2),
                    } for seg in ["power_user", "casual_browser", "price_sensitive", "early_adopter", "enterprise_user"]
                }} for w in range(1, 5)
            ],
        }


# ── Serve Frontend ─────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, 'index.html'))
