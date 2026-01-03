# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path

def create_features(input_path='data/processed/master_disaster_table.csv'):
    """
    Create features for disaster priority prediction
    """
    df = pd.read_csv(input_path)
    print(f"🔧 Engineering features for {len(df):,} disaster events...")
    
    # 1. Create TARGET VARIABLE: Priority Level (High/Medium/Low)
    # Based on deaths per 100k population
    df['Deaths_per_100k'] = df['Deaths_per_100k'].fillna(0)
    
    # Define priority thresholds (adjustable)
    conditions = [
        df['Deaths_per_100k'] >= 10,      # High: ≥10 deaths per 100k
        df['Deaths_per_100k'] >= 1,       # Medium: 1-10 deaths per 100k
        df['Deaths_per_100k'] < 1         # Low: <1 death per 100k
    ]
    choices = ['High', 'Medium', 'Low']
    df['Priority'] = np.select(conditions, choices, default='Low')
    
    # 2. Temporal features
    df['Month'] = df['Start Month'].fillna(1).astype(int)
    df['Season'] = df['Month'].apply(lambda m: 'Winter' if m in [12,1,2] else
                                              'Spring' if m in [3,4,5] else
                                              'Summer' if m in [6,7,8] else 'Fall')
    
    # 3. Disaster type grouping (using existing Disaster Type column)
    disaster_type_map = {
        'Flood': ['Flood', 'Flash flood', 'Coastal flood', 'River flood'],
        'Storm': ['Tropical cyclone', 'Storm', 'Tornado', 'Winter storm', 'Extreme temperature'],
        'Earthquake': ['Earthquake', 'Tsunami', 'Volcanic activity'],
        'Drought': ['Drought'],
        'Other': []  # Everything else
    }
    
    # Create binary columns for each group
    for group, types in disaster_type_map.items():
        if types:  # Skip 'Other' for now
            # Check if any of these types exist in our data
            existing_types = [t for t in types if t in df['Disaster Type'].unique()]
            if existing_types:
                df[f'Is_{group}'] = df['Disaster Type'].isin(existing_types).astype(int)
            else:
                df[f'Is_{group}'] = 0
    
    # 4. Impact severity features
    df['Has_Casualties'] = (df['Total Deaths'].fillna(0) > 0).astype(int)
    df['Has_Affected'] = (df['Total Affected'].fillna(0) > 0).astype(int)
    
    # 5. Population features
    df['Log_Population'] = np.log1p(df['Population'].fillna(0))
    
    # 6. Historical frequency features
    df = df.sort_values(['ISO', 'Start Year', 'Start Month', 'Start Day'])
    df['Country_Disaster_Count'] = df.groupby('ISO').cumcount() + 1
    
    # 7. Time since last disaster (in days approximation)
    df['Start_Date'] = pd.to_datetime(df[['Start Year', 'Start Month', 'Start Day']].fillna(1).astype(str).agg('-'.join, axis=1), errors='coerce')
    df['Days_Since_Last'] = df.groupby('ISO')['Start_Date'].diff().dt.days.fillna(365)  # Default 1 year
    
    print(f" Features created: {len(df.columns)} total columns")
    print(f"\n Target distribution (Priority):")
    print(df['Priority'].value_counts())
    
    return df

def save_features(df, output_path='data/processed/features_engineered.csv'):
    """Save feature-engineered dataset"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Select final feature columns for modeling
    feature_cols = [
        'ISO', 'Country', 'Start Year', 'Month', 'Season',
        'Is_Flood', 'Is_Storm', 'Is_Earthquake', 'Is_Drought',
        'Has_Casualties', 'Has_Affected', 'Log_Population',
        'Country_Disaster_Count', 'Days_Since_Last',
        'Total Deaths', 'Total Affected',  # Raw targets
        'Deaths_per_100k', 'Affected_pct',  # Normalized targets
        'Priority'  # Classification target
    ]
    
    # Keep only existing columns
    existing_cols = [col for col in feature_cols if col in df.columns]
    final_df = df[existing_cols]
    
    final_df.to_csv(output_path, index=False)
    print(f"\n Features saved to: {output_path}")
    print(f"  Features: {len(existing_cols)} columns")
    print(f"  Samples: {len(final_df):,} rows")
    
    return final_df

if __name__ == "__main__":
    # Engineer features
    features_df = create_features()
    final_df = save_features(features_df)
    
    # Show feature summary
    print("\nFEATURE SUMMARY:")
    print("Categorical features:")
    print(f"  - Season: {features_df['Season'].nunique()} categories")
    disaster_flags = [col for col in features_df.columns if col.startswith('Is_')]
    print(f"  - Disaster type flags: {len(disaster_flags)} binary features")
    
    print("\nNumerical features:")
    num_features = final_df.select_dtypes(include=[np.number]).columns.tolist()
    num_features = [f for f in num_features if f not in ['Start Year', 'Month', 'Total Deaths', 'Total Affected']]
    print(f"  - {len(num_features)} features: {', '.join(num_features[:8])}")
    
    print("\nSample priorities by disaster type:")
    for flag in disaster_flags:
        if flag in final_df.columns:
            subset = final_df[final_df[flag] == 1]
            if len(subset) > 0:
                print(f"  {flag}: {subset['Priority'].value_counts().to_dict()}")