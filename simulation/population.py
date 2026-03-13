"""
Synthetic Population Engine — Layer 1
Generates diverse user profiles with realistic distributions for product impact simulation.
"""
from __future__ import annotations

import json
import os
import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ── Segment definitions ────────────────────────────────────────────
SEGMENTS = ['power_user', 'casual_browser', 'price_sensitive', 'early_adopter', 'enterprise_user']

SEGMENT_DISTRIBUTIONS = {
    'power_user': {
        'weight': 0.20,
        'age_mean': 30, 'age_std': 6,
        'income_weights': {'low': 0.10, 'mid': 0.40, 'high': 0.50},
        'usage_weights': {'daily': 0.80, 'weekly': 0.15, 'occasional': 0.05},
        'tech_savviness_range': (7, 10),
        'price_sensitivity_range': (1, 5),
        'feature_adoption_mean': 0.7, 'feature_adoption_std': 0.12,
        'subscription_weights': {'free': 0.05, 'basic': 0.25, 'premium': 0.70},
        'churn_baseline_range': (0.01, 0.04),
        'engagement_mean': 85, 'engagement_std': 8,
        'satisfaction_mean': 80, 'satisfaction_std': 10,
    },
    'casual_browser': {
        'weight': 0.25,
        'age_mean': 35, 'age_std': 12,
        'income_weights': {'low': 0.30, 'mid': 0.50, 'high': 0.20},
        'usage_weights': {'daily': 0.05, 'weekly': 0.35, 'occasional': 0.60},
        'tech_savviness_range': (2, 6),
        'price_sensitivity_range': (3, 7),
        'feature_adoption_mean': 0.15, 'feature_adoption_std': 0.08,
        'subscription_weights': {'free': 0.70, 'basic': 0.25, 'premium': 0.05},
        'churn_baseline_range': (0.05, 0.10),
        'engagement_mean': 30, 'engagement_std': 15,
        'satisfaction_mean': 55, 'satisfaction_std': 15,
    },
    'price_sensitive': {
        'weight': 0.25,
        'age_mean': 28, 'age_std': 8,
        'income_weights': {'low': 0.55, 'mid': 0.35, 'high': 0.10},
        'usage_weights': {'daily': 0.30, 'weekly': 0.45, 'occasional': 0.25},
        'tech_savviness_range': (3, 7),
        'price_sensitivity_range': (7, 10),
        'feature_adoption_mean': 0.35, 'feature_adoption_std': 0.15,
        'subscription_weights': {'free': 0.50, 'basic': 0.40, 'premium': 0.10},
        'churn_baseline_range': (0.05, 0.08),
        'engagement_mean': 55, 'engagement_std': 18,
        'satisfaction_mean': 50, 'satisfaction_std': 18,
    },
    'early_adopter': {
        'weight': 0.15,
        'age_mean': 26, 'age_std': 5,
        'income_weights': {'low': 0.15, 'mid': 0.45, 'high': 0.40},
        'usage_weights': {'daily': 0.60, 'weekly': 0.30, 'occasional': 0.10},
        'tech_savviness_range': (8, 10),
        'price_sensitivity_range': (2, 6),
        'feature_adoption_mean': 0.8, 'feature_adoption_std': 0.10,
        'subscription_weights': {'free': 0.10, 'basic': 0.30, 'premium': 0.60},
        'churn_baseline_range': (0.03, 0.06),
        'engagement_mean': 75, 'engagement_std': 12,
        'satisfaction_mean': 72, 'satisfaction_std': 12,
    },
    'enterprise_user': {
        'weight': 0.15,
        'age_mean': 40, 'age_std': 8,
        'income_weights': {'low': 0.05, 'mid': 0.30, 'high': 0.65},
        'usage_weights': {'daily': 0.50, 'weekly': 0.40, 'occasional': 0.10},
        'tech_savviness_range': (5, 9),
        'price_sensitivity_range': (1, 4),
        'feature_adoption_mean': 0.45, 'feature_adoption_std': 0.15,
        'subscription_weights': {'free': 0.02, 'basic': 0.18, 'premium': 0.80},
        'churn_baseline_range': (0.01, 0.03),
        'engagement_mean': 70, 'engagement_std': 10,
        'satisfaction_mean': 75, 'satisfaction_std': 8,
    },
}


def _weighted_choice(options: dict) -> str:
    """Pick from a dict of {option: probability_weight}."""
    keys = list(options.keys())
    weights = list(options.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def generate_user(user_id: int, segment: str) -> dict:
    """Generate a single user profile for the given segment."""
    cfg = SEGMENT_DISTRIBUTIONS[segment]

    age = int(_clamp(np.random.normal(cfg['age_mean'], cfg['age_std']), 18, 65))
    income_tier = _weighted_choice(cfg['income_weights'])
    usage_frequency = _weighted_choice(cfg['usage_weights'])
    tech_savviness = random.randint(*cfg['tech_savviness_range'])
    price_sensitivity = random.randint(*cfg['price_sensitivity_range'])
    feature_adoption_rate = round(_clamp(
        np.random.normal(cfg['feature_adoption_mean'], cfg['feature_adoption_std']), 0.0, 1.0
    ), 3)
    subscription_tier = _weighted_choice(cfg['subscription_weights'])
    churn_risk_baseline = round(random.uniform(*cfg['churn_baseline_range']), 4)
    engagement_score = int(_clamp(np.random.normal(cfg['engagement_mean'], cfg['engagement_std']), 0, 100))
    satisfaction_score = int(_clamp(np.random.normal(cfg['satisfaction_mean'], cfg['satisfaction_std']), 0, 100))

    return {
        'user_id': f'USR-{user_id:05d}',
        'name': fake.name(),
        'email': fake.email(),
        'age': age,
        'income_tier': income_tier,
        'usage_frequency': usage_frequency,
        'tech_savviness': tech_savviness,
        'price_sensitivity': price_sensitivity,
        'feature_adoption_rate': feature_adoption_rate,
        'subscription_tier': subscription_tier,
        'churn_risk_baseline': churn_risk_baseline,
        'engagement_score': engagement_score,
        'satisfaction_score': satisfaction_score,
        'segment_label': segment,
    }


def generate_population(n: int = 2000) -> list[dict]:
    """
    Generate a synthetic population with stratified segment representation.
    Each segment gets at least 10% of the population.
    """
    population = []
    user_id = 1

    # Assign users per segment based on weights (minimum 10%)
    segment_counts = {}
    for seg, cfg in SEGMENT_DISTRIBUTIONS.items():
        count = max(int(n * cfg['weight']), int(n * 0.10))
        segment_counts[seg] = count

    # Adjust to exactly n
    total = sum(segment_counts.values())
    if total < n:
        # Add remainder to the largest segment
        largest = max(segment_counts, key=segment_counts.get)
        segment_counts[largest] += n - total
    elif total > n:
        largest = max(segment_counts, key=segment_counts.get)
        segment_counts[largest] -= total - n

    for segment, count in segment_counts.items():
        for _ in range(count):
            population.append(generate_user(user_id, segment))
            user_id += 1

    random.shuffle(population)
    return population


def validate_population(population: list[dict]) -> dict:
    """Validate population distributions and return stats."""
    df = pd.DataFrame(population)
    stats = {
        'total_users': len(df),
        'segments': {},
    }
    for seg in SEGMENTS:
        seg_df = df[df['segment_label'] == seg]
        pct = len(seg_df) / len(df) * 100
        stats['segments'][seg] = {
            'count': len(seg_df),
            'percentage': round(pct, 1),
            'avg_age': round(seg_df['age'].mean(), 1),
            'avg_engagement': round(seg_df['engagement_score'].mean(), 1),
            'avg_satisfaction': round(seg_df['satisfaction_score'].mean(), 1),
            'avg_churn_risk': round(seg_df['churn_risk_baseline'].mean(), 4),
            'avg_price_sensitivity': round(seg_df['price_sensitivity'].mean(), 1),
        }
    return stats


def save_population(population: list[dict], output_dir: str = 'data'):
    """Save population to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'population.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(population, f, indent=2)
    print(f"  Saved {len(population)} users to {json_path}")

    csv_path = os.path.join(output_dir, 'population.csv')
    pd.DataFrame(population).to_csv(csv_path, index=False)
    print(f"  Saved CSV to {csv_path}")

    return json_path, csv_path


if __name__ == '__main__':
    print("=" * 60)
    print("  SYNTHETIC POPULATION ENGINE")
    print("=" * 60)

    pop = generate_population(2000)
    stats = validate_population(pop)

    print(f"\n  Generated {stats['total_users']} users across {len(stats['segments'])} segments:\n")
    for seg, s in stats['segments'].items():
        print(f"  {seg:20s}  {s['count']:5d} users ({s['percentage']:5.1f}%)  "
              f"engagement={s['avg_engagement']:.0f}  churn={s['avg_churn_risk']:.3f}")

    save_population(pop)

    # Save stats for dashboard
    os.makedirs('data', exist_ok=True)
    with open('data/population_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\n  Done! Population ready for simulation.")
