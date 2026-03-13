"""
Scalable Approximation Model — Layer 4
PyTorch MLP + rule-based fallback for generalizing LLM predictions to full population.
"""
from __future__ import annotations

import json
import os
import numpy as np
from typing import Optional

# ── Rule-Based Fallback (coded FIRST per plan) ─────────────────────

def rule_based_predict(user: dict, feature_signal: dict) -> dict:
    """
    Approximate impact using segment median deltas scaled by user sensitivity.
    This is the primary fallback when training data < 30 samples.
    """
    segment = user['segment_label']
    direction_map = feature_signal.get('direction_by_segment', {})
    direction = direction_map.get(segment, 'unknown')
    magnitude = feature_signal.get('magnitude_estimate', 0.5)
    change_type = feature_signal.get('change_type', 'policy')

    dir_mult = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.1, 'unknown': 0.0}.get(direction, 0.0)

    # Segment-specific median deltas (calibrated from typical LLM outputs)
    segment_medians = {
        'power_user': {'engagement': 5, 'churn': -0.02, 'satisfaction': 3},
        'casual_browser': {'engagement': -3, 'churn': 0.03, 'satisfaction': -5},
        'price_sensitive': {'engagement': -8, 'churn': 0.08, 'satisfaction': -12},
        'early_adopter': {'engagement': 7, 'churn': -0.01, 'satisfaction': 5},
        'enterprise_user': {'engagement': 2, 'churn': -0.01, 'satisfaction': 2},
    }

    medians = segment_medians.get(segment, {'engagement': 0, 'churn': 0, 'satisfaction': 0})

    # Scale by user's individual sensitivity
    price_factor = user['price_sensitivity'] / 10.0
    adoption_factor = user['feature_adoption_rate']

    if change_type in ['pricing', 'removal']:
        scale = price_factor
    elif change_type in ['new_feature', 'performance']:
        scale = adoption_factor
    else:
        scale = 0.5

    delta_engagement = round(dir_mult * medians['engagement'] * magnitude * scale * 2 + np.random.normal(0, 1.5), 2)
    delta_churn = round(dir_mult * medians['churn'] * magnitude * scale * 2 + np.random.normal(0, 0.005), 4)
    delta_satisfaction = round(dir_mult * medians['satisfaction'] * magnitude * scale * 2 + np.random.normal(0, 1.5), 2)

    # Clamp
    delta_engagement = max(-50, min(50, delta_engagement))
    delta_churn = max(-0.5, min(0.5, delta_churn))
    delta_satisfaction = max(-50, min(50, delta_satisfaction))

    new_churn = user['churn_risk_baseline'] + delta_churn
    will_churn = new_churn > 0.12

    return {
        'user_id': user['user_id'],
        'segment': user['segment_label'],
        'delta_engagement': delta_engagement,
        'delta_churn_risk': delta_churn,
        'delta_satisfaction': delta_satisfaction,
        'will_churn': will_churn,
        'new_engagement': max(0, min(100, user['engagement_score'] + delta_engagement)),
        'new_satisfaction': max(0, min(100, user['satisfaction_score'] + delta_satisfaction)),
        'new_churn_risk': max(0, min(1, new_churn)),
    }


# ── PyTorch MLP ────────────────────────────────────────────────────

def train_and_predict(training_data: dict, population: list[dict],
                       feature_signal: dict, epochs: int = 200) -> list[dict]:
    """
    Train a PyTorch MLP on LLM simulation data, then predict for full population.
    Falls back to rule-based if training data is insufficient.
    """
    X_train = np.array(training_data['X'])
    Y_train = np.array(training_data['Y'])

    if len(X_train) < 30:
        print(f"  [Fallback] Only {len(X_train)} training samples — using rule-based approximation")
        return predict_rule_based_all(population, feature_signal)

    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        import joblib

        print(f"  Training MLP on {len(X_train)} samples...")

        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Save scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')

        # Split 80/20
        n_train = int(0.8 * len(X_scaled))
        X_tr, X_val = X_scaled[:n_train], X_scaled[n_train:]
        Y_tr, Y_val = Y_train[:n_train], Y_train[n_train:]

        X_tr_t = torch.FloatTensor(X_tr)
        Y_tr_t = torch.FloatTensor(Y_tr)
        X_val_t = torch.FloatTensor(X_val)
        Y_val_t = torch.FloatTensor(Y_val)

        # Model
        model = nn.Sequential(
            nn.Linear(18, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_tr_t)
            loss = criterion(pred, Y_tr_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = criterion(val_pred, Y_val_t)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                print(f"  Epoch {epoch+1}/{epochs}  train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}")

        # Save model
        torch.save(model.state_dict(), 'models/approximator.pth')

        # Full population inference
        print(f"  Running inference on {len(population)} users...")
        return predict_with_model(model, scaler, population, feature_signal)

    except ImportError:
        print("  [No PyTorch] Falling back to rule-based approximation")
        return predict_rule_based_all(population, feature_signal)


def predict_with_model(model, scaler, population: list[dict], feature_signal: dict) -> list[dict]:
    """Use trained model to predict for all users."""
    import torch

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
    results = []

    X_all = []
    for user in population:
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
            change_type_map.get(feature_signal.get('change_type', 'policy'), 0.5),
            feature_signal.get('magnitude_estimate', 0.5),
            feature_signal.get('confidence', 0.5),
            1.0 if 'engagement' in affected else 0.0,
            1.0 if 'churn_risk' in affected else 0.0,
            1.0 if 'satisfaction' in affected else 0.0,
        ]
        X_all.append(x)

    X_scaled = scaler.transform(np.array(X_all))
    X_tensor = torch.FloatTensor(X_scaled)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    for i, user in enumerate(population):
        de = float(max(-50, min(50, predictions[i][0])))
        dc = float(max(-0.5, min(0.5, predictions[i][1])))
        ds = float(max(-50, min(50, predictions[i][2])))
        new_churn = user['churn_risk_baseline'] + dc

        results.append({
            'user_id': user['user_id'],
            'segment': user['segment_label'],
            'delta_engagement': round(de, 2),
            'delta_churn_risk': round(dc, 4),
            'delta_satisfaction': round(ds, 2),
            'will_churn': new_churn > 0.12,
            'new_engagement': round(max(0, min(100, user['engagement_score'] + de)), 1),
            'new_satisfaction': round(max(0, min(100, user['satisfaction_score'] + ds)), 1),
            'new_churn_risk': round(max(0, min(1, new_churn)), 4),
        })

    return results


def predict_rule_based_all(population: list[dict], feature_signal: dict) -> list[dict]:
    """Rule-based prediction for all users."""
    return [rule_based_predict(user, feature_signal) for user in population]


# ── Segment Aggregation ───────────────────────────────────────────

def compute_segment_aggregates(predictions: list[dict]) -> dict:
    """Compute segment-level aggregate statistics from predictions."""
    from collections import defaultdict

    by_segment = defaultdict(list)
    for p in predictions:
        by_segment[p['segment']].append(p)

    aggregates = {}
    for segment, preds in by_segment.items():
        de = [p['delta_engagement'] for p in preds]
        dc = [p['delta_churn_risk'] for p in preds]
        ds = [p['delta_satisfaction'] for p in preds]

        aggregates[segment] = {
            'count': len(preds),
            'engagement': {
                'mean': round(float(np.mean(de)), 2),
                'std': round(float(np.std(de)), 2),
                'p10': round(float(np.percentile(de, 10)), 2),
                'p90': round(float(np.percentile(de, 90)), 2),
            },
            'churn_risk': {
                'mean': round(float(np.mean(dc)), 4),
                'std': round(float(np.std(dc)), 4),
                'p10': round(float(np.percentile(dc, 10)), 4),
                'p90': round(float(np.percentile(dc, 90)), 4),
            },
            'satisfaction': {
                'mean': round(float(np.mean(ds)), 2),
                'std': round(float(np.std(ds)), 2),
                'p10': round(float(np.percentile(ds, 10)), 2),
                'p90': round(float(np.percentile(ds, 90)), 2),
            },
            'churn_count': sum(1 for p in preds if p['will_churn']),
            'churn_rate': round(sum(1 for p in preds if p['will_churn']) / len(preds) * 100, 1),
        }

    return aggregates


def compute_gini(values: list[float]) -> float:
    """Compute Gini coefficient for inequality measurement."""
    arr = np.array(values)
    arr = arr - arr.min()  # Shift to non-negative
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float(round((2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n, 4))


# ── CLI Test ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  APPROXIMATION MODEL — Test")
    print("=" * 60)

    with open('data/population.json', 'r') as f:
        population = json.load(f)

    feature_signal = {
        'feature_name': 'Price increase 15%',
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

    # Use rule-based (no training data in this test)
    predictions = predict_rule_based_all(population, feature_signal)
    aggregates = compute_segment_aggregates(predictions)

    print(f"\n  Predictions for {len(predictions)} users:")
    for seg, agg in aggregates.items():
        print(f"\n  {seg} ({agg['count']} users):")
        print(f"    Engagement delta: {agg['engagement']['mean']:+.1f} ± {agg['engagement']['std']:.1f}")
        print(f"    Churn risk delta: {agg['churn_risk']['mean']:+.4f} ± {agg['churn_risk']['std']:.4f}")
        print(f"    Satisfaction delta: {agg['satisfaction']['mean']:+.1f} ± {agg['satisfaction']['std']:.1f}")
        print(f"    Will churn: {agg['churn_count']} ({agg['churn_rate']}%)")

    # Gini
    sat_deltas = [p['delta_satisfaction'] for p in predictions]
    gini = compute_gini(sat_deltas)
    print(f"\n  Gini coefficient (satisfaction): {gini:.4f}")

    # Save
    with open('data/population_with_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    with open('data/segment_aggregates.json', 'w') as f:
        json.dump(aggregates, f, indent=2)
    print("\n  Saved predictions and aggregates.")
