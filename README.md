# AI Sandbox for Product Impact Analysis

**Hack-Nocturne '26** — Team: Sameeksha KC · Sanat Shirwaicar · Shama Hegde

> Simulate how product decisions (pricing changes, feature launches, UX shifts) impact different user segments **before** real-world deployment.

## 🏗️ Architecture

A 5-layer pipeline — each layer independently buildable and testable:

| Layer | Component | Purpose | Tech |
|-------|-----------|---------|------|
| L1 | Synthetic Population Engine | Generate diverse user profiles | Python, NumPy, Faker |
| L2 | Feature Interpretation Layer | Parse product changes to structured signals | Gemini API, Pydantic |
| L3 | High-Fidelity Simulation | Deep behavioral reasoning on sample users | Gemini API, async |
| L4 | Scalable Approximation Model | Generalize to full population | PyTorch MLP |
| L5 | Impact Analytics Dashboard | Visualize segment-level outcomes | FastAPI + Vanilla JS |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 3. Generate population (if not already generated)
python simulation/population.py

# 4. Start server
python -m uvicorn backend.main:app --reload --port 8000

# 5. Open dashboard
# Navigate to http://127.0.0.1:8000
```

## 🔗 Requestly Integration

This API is fully instrumented for [Requestly](https://requestly.io/) testing:

| Feature | How |
|---------|-----|
| **Request Tracing** | `X-Request-Id` + `X-Processing-Time-Ms` headers on every response |
| **Mock Mode** | Send `X-Requestly-Mock: true` header for pre-computed results |
| **Auto Collection** | `GET /api/requestly-collection` — ready-to-import API collection |
| **Request Log** | `GET /api/request-log` — live API traffic monitor |
| **Env Switching** | `{{BASE_URL}}` variable for one-click localhost ↔ deployed |

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/population/segments` | Segment distribution stats |
| `POST` | `/simulate` | Run a product change simulation |
| `GET` | `/results/latest` | Latest simulation results |
| `GET` | `/results/history` | All past simulations |
| `GET` | `/api/requestly-collection` | Download Requestly collection |
| `GET` | `/api/request-log` | View API request log |
| `GET` | `/api/mock-data/{scenario}` | Pre-computed mock data |

## 📁 Project Structure

```
├── backend/
│   └── main.py              # FastAPI server + Requestly middleware
├── simulation/
│   ├── population.py        # Synthetic population engine (L1)
│   ├── feature_interpreter.py # LLM feature parser (L2)
│   └── hifi_simulator.py    # Behavioral simulation (L3)
├── models/
│   └── approximator.py      # PyTorch MLP + rule-based fallback (L4)
├── frontend/
│   └── index.html           # Interactive dashboard (L5)
├── data/                    # Generated data (gitignored)
├── requirements.txt
└── .env.example
```

## 🎯 Demo Script

1. Open dashboard → show 5 segments, 2000 users
2. Simulate: *"Increase premium price by 15% and remove free tier export"*
3. Point to Segment Impact Grid: *"price_sensitive users show +0.4 churn risk delta"*
4. Show Gini score: *"0.41 — this change harms low-income segments disproportionately"*
5. Run second simulation: *"Adding AI-powered search for all tiers"* — contrast the two
