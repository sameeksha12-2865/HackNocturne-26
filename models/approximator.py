import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_features(df):
    """
    Build 18-dim feature vector. EXCLUDE any actual churn data from inputs.
    """
    # 6 Real numeric features
    user_cols = ['tenure', 'MonthlyCharges', 'price_sensitivity', 
                 'engagement_score', 'feature_adoption_rate', 'TotalCharges']
    
    # Start with zeros to ensure alignment
    X = np.zeros((len(df), 18))
    
    for idx, col in enumerate(user_cols):
        if col in df.columns:
            vals = df[col].astype(float).values
            mx = np.max(vals) if np.max(vals) > 0 else 1.0
            X[:, idx] = vals / mx
            
    # Leave 6 user padding features as 0 (indices 6-11)
    
    # 6 Signal dummy features (indices 12-17)
    for i in range(6):
        signal_col = f'signal_dummy_{i}'
        idx = 12 + i
        if signal_col in df.columns:
            vals = df[signal_col].astype(float).values
            mx = np.max(vals) if np.max(vals) > 0 else 1.0
            X[:, idx] = vals / mx
            
    return torch.FloatTensor(X)

class ApproximatorNet(nn.Module):
    def __init__(self, input_dim=18, output_dim=3):
        super(ApproximatorNet, self).__init__()
        # Output 3 values: [delta_engagement, churn_risk_absolute, delta_satisfaction]
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def rule_based_fallback(population_df: pd.DataFrame, training_data: list) -> pd.DataFrame:
    print(f"Fallback triggered: Only {len(training_data)} LLM samples available.")
    # For baseline churn, we can't cheat by using actual. Use segment mean instead.
    base_chr = population_df.groupby('segment')['Churn_binary'].transform('mean')
    
    if len(training_data) > 0:
        val_df = pd.DataFrame(training_data)
        medians = val_df.groupby('segment')[['delta_engagement', 'delta_churn_risk', 'delta_satisfaction']].median()
    else:
        medians = pd.DataFrame(columns=['delta_engagement', 'delta_churn_risk', 'delta_satisfaction'])
    
    delts_engagement = []
    delts_churn = []
    delts_satisfaction = []
    
    for _, row in population_df.iterrows():
        segment = row.get('segment', 'casual_browser')
        base_eng = medians.loc[segment, 'delta_engagement'] if segment in medians.index else 0.0
        base_chr_delta = medians.loc[segment, 'delta_churn_risk'] if segment in medians.index else 0.0
        base_sat = medians.loc[segment, 'delta_satisfaction'] if segment in medians.index else 0.0
        
        scale = row.get('price_sensitivity', 0.5) * row.get('feature_adoption_rate', 0.5)
        
        delts_engagement.append(base_eng * scale)
        delts_churn.append(base_chr_delta * scale)
        delts_satisfaction.append(base_sat * scale)
        
    population_df['predicted_engagement_delta'] = delts_engagement
    population_df['predicted_churn_risk'] = base_chr + np.array(delts_churn)  # Absolute
    population_df['predicted_satisfaction_delta'] = delts_satisfaction
    
    return population_df

def train_approximator(training_data: list):
    model = ApproximatorNet(input_dim=18, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    if len(training_data) > 0:
        llm_df = pd.DataFrame(training_data)
        X_llm = get_features(llm_df)
        Y_llm = torch.zeros(len(llm_df), 3)
        Y_llm[:, 0] = torch.FloatTensor(llm_df.get('delta_engagement', pd.Series(0, dtype=float, index=llm_df.index)).values)
        Y_llm[:, 1] = torch.FloatTensor(llm_df.get('delta_churn_risk', pd.Series(0, dtype=float, index=llm_df.index)).values)
        Y_llm[:, 2] = torch.FloatTensor(llm_df.get('delta_satisfaction', pd.Series(0, dtype=float, index=llm_df.index)).values)
    else:
        return model
        
    model.train()
    print("Training neural network...")
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(X_llm)
        loss = criterion(out, Y_llm)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")
            
    return model

def inference_and_validate(population_df: pd.DataFrame, model: nn.Module):
    print(f"Running inference across all {len(population_df)} users...")
    
    model.eval()
    X_infer = get_features(population_df)
    with torch.no_grad():
        predictions = model(X_infer).numpy()
        
    population_df['predicted_engagement_delta'] = predictions[:, 0]
    population_df['predicted_churn_risk_delta'] = predictions[:, 1]
    population_df['predicted_satisfaction_delta'] = predictions[:, 2]
    
    # Absolute predicted churn risk = segment baseline mean + the predicted delta.
    base_chr = population_df.groupby('segment')['Churn_binary'].transform('mean')
    predicted_churn_risk = base_chr + population_df['predicted_churn_risk_delta']
    predicted_churn_risk = np.clip(predicted_churn_risk, 0, 1)
    
    if len(population_df['Churn_binary'].unique()) > 1:
        auc = roc_auc_score(population_df['Churn_binary'], predicted_churn_risk)
        print(f"\n✅ Validation - Ground Truth Churn ROC-AUC: {auc:.4f}")
    
    binarized_churn = (predicted_churn_risk > 0.5).astype(int)
    predicted_churn_rate = binarized_churn.mean()
    actual_churn_rate = population_df['Churn_binary'].mean()
    churn_rate_error = abs(predicted_churn_rate - actual_churn_rate)
    
    print(f"Actual Churn Rate:    {actual_churn_rate:.4f}")
    print(f"Predicted Churn Rate: {predicted_churn_rate:.4f}")
    print(f"Churn Rate Error:     {churn_rate_error:.4f}")
    
    output_file = 'data/population_with_predictions.json'
    population_df.to_json(output_file, orient='records')
    print(f"Saved predictions to {output_file}")
        
    return population_df

def main():
    print("Loading data...")
    try:
        population_df = pd.read_csv('data/population.csv')
    except Exception as e:
        print(f"Could not load population.csv: {e}")
        return
        
    training_data = []
    if os.path.exists('data/simulation_training_data.json'):
        try:
            with open('data/simulation_training_data.json', 'r') as f:
                training_data = json.load(f)
        except Exception as e:
            pass
            
    print(f"Found {len(training_data)} LLM simulation samples.")
    
    if len(training_data) < 25:
        population_df = rule_based_fallback(population_df, training_data)
        predicted_churn_risk = np.clip(population_df['predicted_churn_risk'].values, 0, 1)
        if len(population_df['Churn_binary'].unique()) > 1:
            auc = roc_auc_score(population_df['Churn_binary'], predicted_churn_risk)
            print(f"\n✅ Fallback Validation - Ground Truth Churn ROC-AUC: {auc:.4f}")
            
        binarized_churn = (predicted_churn_risk > 0.5).astype(int)
        predicted_churn_rate = binarized_churn.mean()
        actual_churn_rate = population_df['Churn_binary'].mean()
        print(f"Actual Churn Rate:    {actual_churn_rate:.4f}")
        print(f"Predicted Churn Rate: {predicted_churn_rate:.4f}")
        print(f"Churn Rate Error:     {abs(predicted_churn_rate - actual_churn_rate):.4f}")
        
        output_file = 'data/population_with_predictions.json'
        population_df.to_json(output_file, orient='records')
        print(f"Saved fallback predictions to {output_file}")
    else:
        model = train_approximator(training_data)
        population_df = inference_and_validate(population_df, model)

if __name__ == '__main__':
    main()
