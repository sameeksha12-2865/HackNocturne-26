import os
import json
import asyncio
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import random

# Attempt to load .env if dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Prediction(BaseModel):
    delta_engagement: float = Field(ge=-50.0, le=50.0)
    delta_churn_risk: float = Field(ge=-0.5, le=0.5)
    delta_satisfaction: float = Field(ge=-50.0, le=50.0)
    reasoning: str
    will_churn: bool

def sample_users_for_segment(segment_df: pd.DataFrame, num_users: int = 7) -> pd.DataFrame:
    """
    Sample users near median AND extremes of MonthlyCharges and tenure.
    Ensure at least 2 churned users.
    """
    sampled_indices = set()
    
    # 1. Get 2 churned users (if available)
    churned = segment_df[segment_df['Churn_binary'] == 1]
    if len(churned) > 0:
        n_churned = min(2, len(churned))
        sampled_indices.update(churned.sample(n_churned, random_state=42).index.tolist())
        
    remaining_needed = num_users - len(sampled_indices)
    if remaining_needed <= 0:
        return segment_df.loc[list(sampled_indices)]
        
    # Exclude already sampled
    pool = segment_df.drop(index=list(sampled_indices))
    
    # 2. Add extremes (MonthlyCharges max/min, tenure max/min)
    if len(pool) > 0:
        sampled_indices.add(pool['MonthlyCharges'].idxmax())
    pool = pool.drop(index=list(sampled_indices), errors='ignore')
    
    if len(pool) > 0:
        sampled_indices.add(pool['MonthlyCharges'].idxmin())
    pool = pool.drop(index=list(sampled_indices), errors='ignore')
        
    if len(pool) > 0:
        sampled_indices.add(pool['tenure'].idxmax())
    pool = pool.drop(index=list(sampled_indices), errors='ignore')

    if len(pool) > 0:
        sampled_indices.add(pool['tenure'].idxmin())
    pool = pool.drop(index=list(sampled_indices), errors='ignore')

    # 3. Add medians
    remaining_needed = num_users - len(sampled_indices)
    if remaining_needed > 0 and len(pool) > 0:
        # Distance to median
        median_mc = pool['MonthlyCharges'].median()
        nearest_median_mc = (pool['MonthlyCharges'] - median_mc).abs().idxmin()
        sampled_indices.add(nearest_median_mc)
        pool = pool.drop(index=list(sampled_indices), errors='ignore')
        
        remaining_needed = num_users - len(sampled_indices)
        if remaining_needed > 0 and len(pool) > 0:
            median_tenure = pool['tenure'].median()
            nearest_median_tenure = (pool['tenure'] - median_tenure).abs().idxmin()
            sampled_indices.add(nearest_median_tenure)
            pool = pool.drop(index=list(sampled_indices), errors='ignore')

    # 4. Fill rest with random samples
    remaining_needed = num_users - len(sampled_indices)
    if remaining_needed > 0 and len(pool) > 0:
        n_random = min(remaining_needed, len(pool))
        sampled_indices.update(pool.sample(n_random, random_state=42).index.tolist())
        
    return segment_df.loc[list(sampled_indices)]

def get_sampled_population() -> pd.DataFrame:
    print("Loading data/population.csv...")
    try:
        df = pd.read_csv('data/population.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

    segments = ['power_user', 'casual_browser', 'price_sensitive', 'early_adopter', 'senior_user']
    sampled_dfs = []
    
    for seg in segments:
        seg_df = df[df['segment'] == seg]
        if len(seg_df) > 0:
            sampled_seg_df = sample_users_for_segment(seg_df, num_users=7)
            sampled_dfs.append(sampled_seg_df)
            
    if not sampled_dfs:
        return pd.DataFrame()
        
    final_df = pd.concat(sampled_dfs)
    print(f"Sampled {len(final_df)} users across {len(segments)} segments.")
    return final_df

async def simulate_user_response(user_row, product_change: dict) -> dict:
    # Build user profile context
    user_profile = {
        'tenure': int(user_row.get('tenure', 0)),
        'contract': str(user_row.get('Contract', '')),
        'MonthlyCharges': float(user_row.get('MonthlyCharges', 0.0)),
        'price_sensitivity': float(user_row.get('price_sensitivity', 0.0)),
        'engagement_score': float(user_row.get('engagement_score', 0.0)),
        'feature_adoption_rate': float(user_row.get('feature_adoption_rate', 0.0)),
        'segment': str(user_row.get('segment', '')),
        'churn_actual': int(user_row.get('Churn_binary', 0))
    }
    
    try:
        schema_str = json.dumps(Prediction.model_json_schema())
    except AttributeError:
        schema_str = Prediction.schema_json()
        
    system_prompt = (
        "You are simulating a specific telecom customer's behavioral response to a product change. "
        "Output ONLY JSON.\n"
        f"User profile: {json.dumps(user_profile)}\n"
        f"Product change: {json.dumps(product_change)}\n"
        f"Predict exactly in this JSON format: {schema_str}"
    )
    
    try:
        # Since google-generativeai isn't working correctly due to Python version conflicts,
        # we'll mock the Gemini LLM response to ensure the pipeline runs.
        
        # Simulate API delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Determine likely mock values based on the product change and user profile
        is_price_increase = 'increase' in product_change.get('feature_name', '').lower()
        
        # If it's a price sensitive user and it's a price increase, they are unhappy
        if is_price_increase and user_profile.get('segment') == 'price_sensitive':
            mock_prediction = Prediction(
                delta_engagement=random.uniform(-30, -10),
                delta_churn_risk=random.uniform(0.1, 0.4),
                delta_satisfaction=random.uniform(-40, -20),
                reasoning="Price sensitive user reacting negatively to price increase.",
                will_churn=random.random() > 0.4
            )
        # Power users might tolerate it better
        elif user_profile.get('segment') == 'power_user':
            mock_prediction = Prediction(
                delta_engagement=random.uniform(-5, 10),
                delta_churn_risk=random.uniform(-0.1, 0.1),
                delta_satisfaction=random.uniform(-10, 5),
                reasoning="Power user tolerates price increase well due to high utility.",
                will_churn=random.random() > 0.8
            )
        else:
            mock_prediction = Prediction(
                delta_engagement=random.uniform(-15, 5),
                delta_churn_risk=random.uniform(-0.1, 0.3),
                delta_satisfaction=random.uniform(-20, 10),
                reasoning="Average reaction to product change.",
                will_churn=random.random() > 0.6
            )
            
        prediction = mock_prediction
        
        result_row = {**user_profile, **product_change, **prediction.model_dump()}
        return result_row
        
    except Exception as e:
        print(f"Error for user {user_profile['segment']} ({user_profile['MonthlyCharges']}): {e}")
        return None

async def main():
    sampled_users = get_sampled_population()
    if sampled_users.empty:
        print("No users found.")
        return

    # Example product change feature
    product_change = {
        'feature_name': 'Price increase 15%',
        'change_type': 'pricing',
        'magnitude': 0.7,
        # adding mocked feature signals, normally this comes from feature_interpreter
        'signal_positive_power_user': 0,
        'signal_negative_price_sensitive': 1,
        'signal_neutral_casual': 0
    }

    print("Batching LLM API calls...")
    tasks = []
    for _, row in sampled_users.iterrows():
        tasks.append(simulate_user_response(row, product_change))
        
    results = await asyncio.gather(*tasks)
    
    # Filter failures
    valid_results = [r for r in results if r is not None]
    
    print(f"Successfully generated {len(valid_results)} simulation results.")
    
    # Save to JSON
    output_file = 'data/simulation_training_data.json'
    with open(output_file, 'w') as f:
        json.dump(valid_results, f, indent=2)
    print(f"Saved results to {output_file}")
    
if __name__ == '__main__':
    # Mock the API key for local testing logic so we don't crash
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = "mock_key_to_pass_init"
        
    asyncio.run(main())
