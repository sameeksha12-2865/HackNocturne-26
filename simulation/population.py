import json
import numpy as np
import pandas as pd
from faker import Faker
from typing import List, Dict
import os

fake = Faker()
np.random.seed(42)

# ── Segment definitions ────────────────────────────────────────────────────────
# Each segment: (min_pct_of_population, attribute distributions)
SEGMENT_DISTRIBUTIONS = {
    "power_user": {
        "weight": 0.18,
        "age":                  {"dist": "normal",  "mu": 29,  "sigma": 5,    "clip": (18, 50)},
        "income_tier":          {"dist": "choice",  "options": ["mid", "high"],          "p": [0.35, 0.65]},
        "usage_frequency":      {"dist": "choice",  "options": ["daily", "weekly", "occasional"], "p": [0.80, 0.15, 0.05]},
        "tech_savviness":       {"dist": "normal",  "mu": 8.5, "sigma": 1.0,  "clip": (6, 10)},
        "price_sensitivity":    {"dist": "normal",  "mu": 3.5, "sigma": 1.5,  "clip": (1, 10)},
        "feature_adoption_rate":{"dist": "normal",  "mu": 0.78,"sigma": 0.12, "clip": (0.4, 1.0)},
        "subscription_tier":    {"dist": "choice",  "options": ["free","basic","premium"],  "p": [0.05, 0.25, 0.70]},
        "churn_risk_baseline":  {"dist": "normal",  "mu": 0.08,"sigma": 0.04, "clip": (0.01, 0.30)},
        "engagement_score":     {"dist": "normal",  "mu": 82,  "sigma": 10,   "clip": (50, 100)},
        "satisfaction_score":   {"dist": "normal",  "mu": 78,  "sigma": 10,   "clip": (40, 100)},
    },
    "casual_browser": {
        "weight": 0.28,
        "age":                  {"dist": "normal",  "mu": 35,  "sigma": 10,   "clip": (18, 65)},
        "income_tier":          {"dist": "choice",  "options": ["low","mid","high"],      "p": [0.30, 0.55, 0.15]},
        "usage_frequency":      {"dist": "choice",  "options": ["daily","weekly","occasional"], "p": [0.15, 0.45, 0.40]},
        "tech_savviness":       {"dist": "normal",  "mu": 5.0, "sigma": 1.8,  "clip": (1, 10)},
        "price_sensitivity":    {"dist": "normal",  "mu": 5.5, "sigma": 2.0,  "clip": (1, 10)},
        "feature_adoption_rate":{"dist": "normal",  "mu": 0.30,"sigma": 0.12, "clip": (0.05, 0.70)},
        "subscription_tier":    {"dist": "choice",  "options": ["free","basic","premium"],  "p": [0.55, 0.35, 0.10]},
        "churn_risk_baseline":  {"dist": "normal",  "mu": 0.22,"sigma": 0.08, "clip": (0.05, 0.60)},
        "engagement_score":     {"dist": "normal",  "mu": 42,  "sigma": 15,   "clip": (10, 75)},
        "satisfaction_score":   {"dist": "normal",  "mu": 55,  "sigma": 14,   "clip": (20, 85)},
    },
    "price_sensitive": {
        "weight": 0.25,
        "age":                  {"dist": "normal",  "mu": 31,  "sigma": 9,    "clip": (18, 60)},
        "income_tier":          {"dist": "choice",  "options": ["low","mid","high"],      "p": [0.65, 0.30, 0.05]},
        "usage_frequency":      {"dist": "choice",  "options": ["daily","weekly","occasional"], "p": [0.30, 0.40, 0.30]},
        "tech_savviness":       {"dist": "normal",  "mu": 5.5, "sigma": 2.0,  "clip": (1, 10)},
        "price_sensitivity":    {"dist": "normal",  "mu": 8.5, "sigma": 1.0,  "clip": (5, 10)},
        "feature_adoption_rate":{"dist": "normal",  "mu": 0.25,"sigma": 0.10, "clip": (0.05, 0.60)},
        "subscription_tier":    {"dist": "choice",  "options": ["free","basic","premium"],  "p": [0.70, 0.25, 0.05]},
        "churn_risk_baseline":  {"dist": "normal",  "mu": 0.35,"sigma": 0.10, "clip": (0.10, 0.70)},
        "engagement_score":     {"dist": "normal",  "mu": 48,  "sigma": 14,   "clip": (10, 80)},
        "satisfaction_score":   {"dist": "normal",  "mu": 50,  "sigma": 15,   "clip": (15, 80)},
    },
    "early_adopter": {
        "weight": 0.15,
        "age":                  {"dist": "normal",  "mu": 26,  "sigma": 5,    "clip": (18, 45)},
        "income_tier":          {"dist": "choice",  "options": ["low","mid","high"],      "p": [0.15, 0.50, 0.35]},
        "usage_frequency":      {"dist": "choice",  "options": ["daily","weekly","occasional"], "p": [0.65, 0.28, 0.07]},
        "tech_savviness":       {"dist": "normal",  "mu": 9.0, "sigma": 0.8,  "clip": (7, 10)},
        "price_sensitivity":    {"dist": "normal",  "mu": 4.0, "sigma": 1.5,  "clip": (1, 8)},
        "feature_adoption_rate":{"dist": "normal",  "mu": 0.88,"sigma": 0.08, "clip": (0.65, 1.0)},
        "subscription_tier":    {"dist": "choice",  "options": ["free","basic","premium"],  "p": [0.10, 0.30, 0.60]},
        "churn_risk_baseline":  {"dist": "normal",  "mu": 0.12,"sigma": 0.05, "clip": (0.02, 0.35)},
        "engagement_score":     {"dist": "normal",  "mu": 88,  "sigma": 8,    "clip": (60, 100)},
        "satisfaction_score":   {"dist": "normal",  "mu": 82,  "sigma": 9,    "clip": (55, 100)},
    },
    "enterprise_user": {
        "weight": 0.14,
        "age":                  {"dist": "normal",  "mu": 42,  "sigma": 8,    "clip": (28, 60)},
        "income_tier":          {"dist": "choice",  "options": ["low","mid","high"],      "p": [0.02, 0.28, 0.70]},
        "usage_frequency":      {"dist": "choice",  "options": ["daily","weekly","occasional"], "p": [0.70, 0.25, 0.05]},
        "tech_savviness":       {"dist": "normal",  "mu": 6.5, "sigma": 1.8,  "clip": (3, 10)},
        "price_sensitivity":    {"dist": "normal",  "mu": 2.5, "sigma": 1.2,  "clip": (1, 7)},
        "feature_adoption_rate":{"dist": "normal",  "mu": 0.45,"sigma": 0.15, "clip": (0.15, 0.80)},
        "subscription_tier":    {"dist": "choice",  "options": ["free","basic","premium"],  "p": [0.02, 0.18, 0.80]},
        "churn_risk_baseline":  {"dist": "normal",  "mu": 0.06,"sigma": 0.03, "clip": (0.01, 0.20)},
        "engagement_score":     {"dist": "normal",  "mu": 72,  "sigma": 12,   "clip": (40, 100)},
        "satisfaction_score":   {"dist": "normal",  "mu": 70,  "sigma": 12,   "clip": (35, 95)},
    },
}


# ── Attribute sampler ──────────────────────────────────────────────────────────

def _sample_attr(cfg: dict, n: int) -> np.ndarray:
    if cfg["dist"] == "normal":
        vals = np.random.normal(cfg["mu"], cfg["sigma"], n)
        vals = np.clip(vals, cfg["clip"][0], cfg["clip"][1])
        return vals
    elif cfg["dist"] == "choice":
        return np.random.choice(cfg["options"], size=n, p=cfg["p"])
    raise ValueError(f"Unknown dist: {cfg['dist']}")


# ── Single-segment generator ───────────────────────────────────────────────────

def _generate_segment(segment_label: str, n: int, id_offset: int) -> List[Dict]:
    cfg = SEGMENT_DISTRIBUTIONS[segment_label]
    users = []

    ages               = _sample_attr(cfg["age"], n).astype(int)
    income_tiers       = _sample_attr(cfg["income_tier"], n)
    usage_freqs        = _sample_attr(cfg["usage_frequency"], n)
    tech_sav           = np.round(_sample_attr(cfg["tech_savviness"], n), 1)
    price_sens         = np.round(_sample_attr(cfg["price_sensitivity"], n), 1)
    adoption_rates     = np.round(_sample_attr(cfg["feature_adoption_rate"], n), 3)
    sub_tiers          = _sample_attr(cfg["subscription_tier"], n)
    churn_risks        = np.round(_sample_attr(cfg["churn_risk_baseline"], n), 4)
    engagement_scores  = np.round(_sample_attr(cfg["engagement_score"], n), 1)
    satisfaction_scores= np.round(_sample_attr(cfg["satisfaction_score"], n), 1)

    for i in range(n):
        users.append({
            "user_id":              f"u_{id_offset + i:06d}",
            "segment_label":        segment_label,
            "age":                  int(ages[i]),
            "income_tier":          str(income_tiers[i]),
            "usage_frequency":      str(usage_freqs[i]),
            "tech_savviness":       float(tech_sav[i]),
            "price_sensitivity":    float(price_sens[i]),
            "feature_adoption_rate":float(adoption_rates[i]),
            "subscription_tier":    str(sub_tiers[i]),
            "churn_risk_baseline":  float(churn_risks[i]),
            "engagement_score":     float(engagement_scores[i]),
            "satisfaction_score":   float(satisfaction_scores[i]),
            "name":                 fake.name(),
            "email":                fake.email(),
            "locale":               fake.locale(),
            "country":              fake.country(),
        })
    return users


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_population(n: int = 5000, seed: int = 42) -> List[Dict]:
    """
    Generate a stratified synthetic population of n users.
    Each segment gets at least 10% representation.
    Returns list of user dicts.
    """
    np.random.seed(seed)
    fake.seed_instance(seed)

    segments      = list(SEGMENT_DISTRIBUTIONS.keys())
    weights       = np.array([SEGMENT_DISTRIBUTIONS[s]["weight"] for s in segments])
    weights       = weights / weights.sum()          # normalize (should already sum to 1)

    # Enforce minimum 10% per segment
    min_pct   = 0.10
    n_segments= len(segments)
    counts    = np.maximum((weights * n).astype(int), int(n * min_pct))

    # Adjust so total == n (add/subtract from largest segment)
    diff = n - counts.sum()
    largest_idx = int(np.argmax(counts))
    counts[largest_idx] += diff

    all_users  = []
    id_offset  = 0
    for seg, count in zip(segments, counts):
        users     = _generate_segment(seg, int(count), id_offset)
        all_users.extend(users)
        id_offset += count

    # Shuffle so segments aren't contiguous
    np.random.shuffle(all_users)
    # Re-assign sequential IDs after shuffle
    for idx, user in enumerate(all_users):
        user["user_id"] = f"u_{idx:06d}"

    return all_users


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_population(users: List[Dict], tolerance: float = 0.05) -> bool:
    df = pd.DataFrame(users)
    total = len(df)
    print(f"\n{'='*55}")
    print(f"  Population size : {total:,}")
    print(f"{'='*55}")

    all_passed = True
    segment_counts = df["segment_label"].value_counts()

    print("\n  Segment distribution:")
    for seg in SEGMENT_DISTRIBUTIONS:
        actual_pct  = segment_counts.get(seg, 0) / total
        target_pct  = SEGMENT_DISTRIBUTIONS[seg]["weight"]
        status      = "✓" if abs(actual_pct - target_pct) <= tolerance + 0.02 else "✗"
        if status == "✗":
            all_passed = False
        print(f"    {status} {seg:<18} actual={actual_pct:.2%}  target={target_pct:.2%}")

    print("\n  Numerical attribute ranges:")
    checks = {
        "age":                   (18, 65),
        "tech_savviness":        (1,  10),
        "price_sensitivity":     (1,  10),
        "feature_adoption_rate": (0,   1),
        "churn_risk_baseline":   (0,   1),
        "engagement_score":      (0, 100),
        "satisfaction_score":    (0, 100),
    }
    for col, (lo, hi) in checks.items():
        mn, mx = df[col].min(), df[col].max()
        ok = lo <= mn and mx <= hi
        if not ok:
            all_passed = False
        print(f"    {'✓' if ok else '✗'} {col:<28} min={mn:.2f}  max={mx:.2f}")

    print("\n  Subscription tier split:")
    print(df["subscription_tier"].value_counts(normalize=True).to_string())
    print("\n  Income tier split:")
    print(df["income_tier"].value_counts(normalize=True).to_string())

    print(f"\n{'='*55}")
    print(f"  Validation {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    print(f"{'='*55}\n")
    return all_passed


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_population(users: List[Dict], prefix: str = "population"):
    with open(f"{prefix}.json", "w") as f:
        json.dump(users, f, indent=2)
    pd.DataFrame(users).to_csv(f"{prefix}.csv", index=False)
    print(f"  Saved {prefix}.json and {prefix}.csv  ({len(users):,} users)")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sizes = {
        "population_dev":    1000,    # quick iteration
        "population_demo":   5000,    # hackathon demo  ← main one
        "population_stress": 15000,   # stress test
    }

    for prefix, n in sizes.items():
        print(f"\nGenerating {prefix}  (n={n:,}) ...")
        users = generate_population(n=n, seed=42)
        validate_population(users)
        save_population(users, prefix=prefix)

    print("\nDone. Use population_demo.json for all live runs.\n")