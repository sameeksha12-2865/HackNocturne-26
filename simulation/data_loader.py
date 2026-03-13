import pandas as pd
import numpy as np

def assign_segment(row):
    if row['Contract'] == 'Two year' and row['MonthlyCharges'] > 65:
        return 'power_user'
    elif row['tenure'] < 6 and row['MonthlyCharges'] < 40:
        return 'price_sensitive'
    elif row['feature_adoption_rate'] > 0.75:
        return 'early_adopter'
    elif row['SeniorCitizen'] == 1:
        return 'senior_user'
    else:
        return 'casual_browser'

def get_tech_savviness(row):
    if row['InternetService'] == 'Fiber optic' and row['PaperlessBilling'] == 'Yes':
        return 'high'
    elif row['InternetService'] == 'DSL':
        return 'mid'
    else:
        return 'low'

def main():
    print("Loading data/telco_churn.csv...")
    try:
        df = pd.read_csv('data/telco_churn.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Step 1: Clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Step 2: Feature Engineering
    # price_sensitivity: High monthly + month-to-month -> high sensitivity. Normalise 0-1.
    max_charge = df['MonthlyCharges'].max()
    df['price_sensitivity'] = df['MonthlyCharges'] / max_charge
    is_month_to_month = (df['Contract'] == 'Month-to-month').astype(float)
    # Add a bump for month-to-month, capping at 1.0
    df['price_sensitivity'] = (df['price_sensitivity'] * 0.8 + is_month_to_month * 0.2).clip(upper=1.0)

    # churn_risk_baseline
    df['churn_risk_baseline'] = df['Churn_binary']

    # engagement_score: count active streaming services + tenure weight -> 0-100
    tv_active = (df['StreamingTV'] == 'Yes').astype(int)
    movies_active = (df['StreamingMovies'] == 'Yes').astype(int)
    max_tenure = df['tenure'].max()
    if max_tenure == 0:  # Prevent division by zero just in case
        max_tenure = 1
    # 50 points from streaming, 50 points from tenure
    df['engagement_score'] = ((tv_active + movies_active) / 2.0 * 50) + ((df['tenure'] / max_tenure) * 50)

    # feature_adoption_rate: count of 'Yes' add-ons / 4 -> 0-1
    features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    feature_count = sum((df[feat] == 'Yes').astype(int) for feat in features)
    df['feature_adoption_rate'] = feature_count / 4.0

    # subscription_tier
    tier_map = {
        'Month-to-month': 'basic',
        'One year': 'standard',
        'Two year': 'premium'
    }
    df['subscription_tier'] = df['Contract'].map(tier_map)

    # tech_savviness
    df['tech_savviness'] = df.apply(get_tech_savviness, axis=1)

    # satisfaction_score: Long tenure + no churn -> high. Churned -> low. Normalise 0-100.
    base_score = (1 - df['Churn_binary']) * 50  # 50 if not churned, 0 if churned
    tenure_score = (df['tenure'] / max_tenure) * 50
    df['satisfaction_score'] = base_score + tenure_score

    # Step 3: Segment Labels
    df['segment'] = df.apply(assign_segment, axis=1)

    # Output Files
    print("Writing output files: data/population.csv and data/population.json...")
    df.to_csv('data/population.csv', index=False)
    # the user requested 'records' orient for JSON
    df.to_json('data/population.json', orient='records')

    print("Success!")
    print("\n--- Validation Stats ---")
    print("\nSegment counts:")
    print(df['segment'].value_counts())
    
    print("\nNull counts in engineered fields:")
    engineered_cols = ['price_sensitivity', 'churn_risk_baseline', 'engagement_score', 
                       'feature_adoption_rate', 'subscription_tier', 'tech_savviness', 'satisfaction_score']
    print(df[engineered_cols].isnull().sum())
    
    print("\nChurn rate per segment:")
    print(df.groupby('segment')['Churn_binary'].mean())

if __name__ == '__main__':
    main()
