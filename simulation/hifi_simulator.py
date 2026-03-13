"""
High-Fidelity Simulation — Layer 3
Uses LLM to deeply reason about how sample users respond to product changes.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import numpy as np
from typing import Optional


# ── Stratified Sampler ─────────────────────────────────────────────

def stratified_sample(population: list[dict], per_segment: int = 6) -> list[dict]:
    """
    Select representative users per segment: median and extremes of engagement.
    Returns 5 * per_segment users (default: 30 users).
    """
    from collections import defaultdict
    by_segment = defaultdict(list)
    for user in population:
        by_segment[user['segment_label']].append(user)

    sampled = []
    for segment, users in by_segment.items():
        sorted_users = sorted(users, key=lambda u: u['engagement_score'])
        n = len(sorted_users)

        if n <= per_segment:
            sampled.extend(sorted_users)
            continue

        # Pick from extremes and median
        indices = set()
        indices.add(0)                     # lowest engagement
        indices.add(n - 1)                 # highest engagement
        indices.add(n // 2)                # median
        indices.add(n // 4)                # low quartile
        indices.add(3 * n // 4)            # high quartile

        # Fill remaining with random picks
        while len(indices) < per_segment and len(indices) < n:
            indices.add(random.randint(0, n - 1))

        for i in sorted(indices):
            sampled.append(sorted_users[i])

    return sampled


# ── Simulation Prompt Builder ──────────────────────────────────────

SIMULATION_SYSTEM_PROMPT = """You are simulating a specific user's behavioral response to a product change. 
You must reason carefully about how this particular user — given their demographics, tech literacy, price sensitivity, and current engagement — would react.

Output ONLY valid JSON matching this schema:
{
  "delta_engagement": float between -50 and +50,
  "delta_churn_risk": float between -0.5 and +0.5,
  "delta_satisfaction": float between -50 and +50,
  "reasoning": "1-2 sentence explanation",
  "will_churn": true or false
}

Rules:
- delta values represent change from current levels
- A price-sensitive user with high churn risk will react more strongly to pricing changes
- Power users are sticky but care about feature depth
- Consider the interaction between user attributes and the specific change
- Output ONLY the JSON, no markdown"""


def build_user_prompt(user: dict, feature_signal: dict) -> str:
    """Build per-user simulation prompt."""
    user_profile = {
        'age': user['age'],
        'income_tier': user['income_tier'],
        'usage_frequency': user['usage_frequency'],
        'tech_savviness': user['tech_savviness'],
        'price_sensitivity': user['price_sensitivity'],
        'feature_adoption_rate': user['feature_adoption_rate'],
        'subscription_tier': user['subscription_tier'],
        'churn_risk_baseline': user['churn_risk_baseline'],
        'engagement_score': user['engagement_score'],
        'satisfaction_score': user['satisfaction_score'],
        'segment': user['segment_label'],
    }

    return f"""User profile: {json.dumps(user_profile)}

Product change: {json.dumps(feature_signal)}

Predict the behavioral response for this specific user:"""


# ── LLM Simulation (Async) ────────────────────────────────────────

async def simulate_user_async(user: dict, feature_signal: dict, api_key: str,
                               semaphore: asyncio.Semaphore) -> dict:
    """Simulate a single user's response via Gemini API."""
    async with semaphore:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            prompt = build_user_prompt(user, feature_signal)
            response = model.generate_content(
                f"{SIMULATION_SYSTEM_PROMPT}\n\n{prompt}"
            )

            # Parse response
            text = response.text.strip()
            if text.startswith('```'):
                lines = text.split('\n')
                text = '\n'.join(lines[1:-1])

            result = json.loads(text)
            result['user_id'] = user['user_id']
            result['segment'] = user['segment_label']

            # Wait to avoid rate limiting
            await asyncio.sleep(0.5)
            return result

        except Exception as e:
            print(f"  [Sim Error] User {user['user_id']}: {e}")
            return simulate_user_rule_based(user, feature_signal)


async def run_simulation_async(sampled_users: list[dict], feature_signal: dict,
                                api_key: str, max_concurrent: int = 5) -> list[dict]:
    """Run simulation on all sampled users with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        simulate_user_async(user, feature_signal, api_key, semaphore)
        for user in sampled_users
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── Rule-Based Fallback Simulation ─────────────────────────────────

def simulate_user_rule_based(user: dict, feature_signal: dict) -> dict:
    """
    Rule-based simulation fallback — no LLM needed.
    Uses segment direction from feature signal + user's individual sensitivity.
    """
    segment = user['segment_label']
    direction_map = feature_signal.get('direction_by_segment', {})
    direction = direction_map.get(segment, 'unknown')
    magnitude = feature_signal.get('magnitude_estimate', 0.5)
    change_type = feature_signal.get('change_type', 'policy')

    # Direction multiplier
    dir_mult = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.1, 'unknown': 0.0}[direction]

    # User-specific sensitivity scaling
    price_factor = user['price_sensitivity'] / 10.0
    tech_factor = user['tech_savviness'] / 10.0
    adoption_factor = user['feature_adoption_rate']
    engagement_factor = user['engagement_score'] / 100.0

    # Base deltas scaled by magnitude
    base_delta = magnitude * 30  # max ±30 for engagement/satisfaction

    if change_type in ['pricing', 'removal']:
        # Pricing/removal: high price_sensitivity → bigger negative swing
        sensitivity = price_factor * 0.6 + (1 - engagement_factor) * 0.4
        delta_engagement = round(dir_mult * base_delta * sensitivity + np.random.normal(0, 2), 2)
        delta_satisfaction = round(dir_mult * base_delta * sensitivity * 1.2 + np.random.normal(0, 2), 2)
        delta_churn = round(-dir_mult * magnitude * 0.3 * sensitivity + np.random.normal(0, 0.02), 4)
    elif change_type in ['new_feature', 'performance']:
        # New features: tech-savvy and early adopters benefit more
        sensitivity = tech_factor * 0.4 + adoption_factor * 0.4 + engagement_factor * 0.2
        delta_engagement = round(dir_mult * base_delta * sensitivity + np.random.normal(0, 3), 2)
        delta_satisfaction = round(dir_mult * base_delta * sensitivity * 0.8 + np.random.normal(0, 2), 2)
        delta_churn = round(-dir_mult * magnitude * 0.15 * (1 - sensitivity) + np.random.normal(0, 0.01), 4)
    else:
        # UI changes, policy
        sensitivity = 0.5
        delta_engagement = round(dir_mult * base_delta * 0.3 + np.random.normal(0, 3), 2)
        delta_satisfaction = round(dir_mult * base_delta * 0.4 + np.random.normal(0, 3), 2)
        delta_churn = round(-dir_mult * magnitude * 0.1 + np.random.normal(0, 0.02), 4)

    # Clamp values
    delta_engagement = max(-50, min(50, delta_engagement))
    delta_satisfaction = max(-50, min(50, delta_satisfaction))
    delta_churn = max(-0.5, min(0.5, delta_churn))

    # Determine churn
    new_churn_risk = user['churn_risk_baseline'] + delta_churn
    will_churn = new_churn_risk > 0.15

    return {
        'user_id': user['user_id'],
        'segment': user['segment_label'],
        'delta_engagement': delta_engagement,
        'delta_churn_risk': delta_churn,
        'delta_satisfaction': delta_satisfaction,
        'reasoning': f"Rule-based: {change_type} change with {direction} impact on {segment}. "
                     f"User sensitivity factor: price={price_factor:.1f}, tech={tech_factor:.1f}.",
        'will_churn': will_churn,
    }


def run_simulation_sync(sampled_users: list[dict], feature_signal: dict,
                         use_llm: bool = False, api_key: Optional[str] = None) -> list[dict]:
    """
    Run simulation on sampled users.
    Falls back to rule-based if use_llm=False or no API key.
    """
    if use_llm and api_key and api_key != 'your_gemini_api_key_here':
        return asyncio.run(run_simulation_async(sampled_users, feature_signal, api_key))
    else:
        print("  [Rule-Based Mode] Running simulations without LLM")
        return [simulate_user_rule_based(u, feature_signal) for u in sampled_users]


# ── Training Dataset Builder ──────────────────────────────────────

def build_training_dataset(population: list[dict], simulation_results: list[dict],
                            feature_signal: dict) -> dict:
    """
    Build training dataset: X = user features + signal features, Y = deltas.
    Returns dict with 'X', 'Y', 'feature_names', 'target_names'.
    """
    # Map simulation results by user_id
    result_map = {r['user_id']: r for r in simulation_results}

    user_feature_names = [
        'age', 'income_tier_encoded', 'usage_freq_encoded',
        'tech_savviness', 'price_sensitivity', 'feature_adoption_rate',
        'subscription_tier_encoded', 'churn_risk_baseline',
        'engagement_score', 'satisfaction_score',
        'segment_encoded_0', 'segment_encoded_1',
    ]

    signal_feature_names = [
        'change_type_encoded', 'magnitude_estimate', 'confidence',
        'dim_engagement', 'dim_churn_risk', 'dim_satisfaction',
    ]

    # Encoders
    income_map = {'low': 0, 'mid': 0.5, 'high': 1.0}
    usage_map = {'occasional': 0, 'weekly': 0.5, 'daily': 1.0}
    sub_map = {'free': 0, 'basic': 0.5, 'premium': 1.0}
    segment_map = {
        'power_user': (1, 0), 'casual_browser': (0, 0), 'price_sensitive': (0, 1),
        'early_adopter': (1, 1), 'enterprise_user': (0.5, 0.5),
    }
    change_type_map = {
        'pricing': 0.0, 'ui_change': 0.2, 'new_feature': 0.4,
        'removal': 0.6, 'performance': 0.8, 'policy': 1.0,
    }

    affected = feature_signal.get('affected_dimensions', [])

    X, Y = [], []
    for uid, result in result_map.items():
        user = next((u for u in population if u['user_id'] == uid), None)
        if user is None:
            continue

        seg_enc = segment_map.get(user['segment_label'], (0, 0))
        x = [
            user['age'] / 65.0,
            income_map.get(user['income_tier'], 0.5),
            usage_map.get(user['usage_frequency'], 0.5),
            user['tech_savviness'] / 10.0,
            user['price_sensitivity'] / 10.0,
            user['feature_adoption_rate'],
            sub_map.get(user['subscription_tier'], 0.5),
            user['churn_risk_baseline'],
            user['engagement_score'] / 100.0,
            user['satisfaction_score'] / 100.0,
            seg_enc[0], seg_enc[1],
            # Feature signal fields
            change_type_map.get(feature_signal.get('change_type', 'policy'), 0.5),
            feature_signal.get('magnitude_estimate', 0.5),
            feature_signal.get('confidence', 0.5),
            1.0 if 'engagement' in affected else 0.0,
            1.0 if 'churn_risk' in affected else 0.0,
            1.0 if 'satisfaction' in affected else 0.0,
        ]

        y = [
            result['delta_engagement'],
            result['delta_churn_risk'],
            result['delta_satisfaction'],
        ]

        X.append(x)
        Y.append(y)

    return {
        'X': X,
        'Y': Y,
        'feature_names': user_feature_names + signal_feature_names,
        'target_names': ['delta_engagement', 'delta_churn_risk', 'delta_satisfaction'],
    }


# ── CLI Test ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  HIGH-FIDELITY SIMULATOR — Test")
    print("=" * 60)

    # Load population
    with open('data/population.json', 'r') as f:
        population = json.load(f)
    print(f"  Loaded {len(population)} users")

    # Sample
    sampled = stratified_sample(population, per_segment=6)
    print(f"  Sampled {len(sampled)} users for simulation")

    # Mock feature signal
    feature_signal = {
        'feature_name': 'Price increase 15% + remove free export',
        'change_type': 'pricing',
        'affected_dimensions': ['churn_risk', 'satisfaction', 'spend'],
        'direction_by_segment': {
            'power_user': 'neutral',
            'casual_browser': 'negative',
            'price_sensitive': 'negative',
            'early_adopter': 'neutral',
            'enterprise_user': 'neutral',
        },
        'magnitude_estimate': 0.7,
        'confidence': 0.65,
    }

    # Run rule-based simulation
    results = run_simulation_sync(sampled, feature_signal, use_llm=False)
    print(f"\n  Simulation complete: {len(results)} results")

    # Show sample results
    for r in results[:3]:
        print(f"  {r['user_id']} ({r['segment']}): ΔEng={r['delta_engagement']:+.1f}  "
              f"ΔChurn={r['delta_churn_risk']:+.4f}  ΔSat={r['delta_satisfaction']:+.1f}  "
              f"Churn={r['will_churn']}")

    # Build training dataset
    dataset = build_training_dataset(population, results, feature_signal)
    print(f"\n  Training dataset: {len(dataset['X'])} samples × {len(dataset['X'][0])} features → {len(dataset['Y'][0])} targets")

    # Save
    os.makedirs('data', exist_ok=True)
    with open('data/simulation_training_data.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    print("  Saved training data to data/simulation_training_data.json")
