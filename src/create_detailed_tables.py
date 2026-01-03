import pandas as pd
import numpy as np
from pathlib import Path

def create_country_risk_ranking():
    """Table 1: Top 10 Most Vulnerable Countries"""
    # Load data
    df = pd.read_csv('data/processed/master_disaster_table.csv')
    
    # Calculate risk metrics per country
    country_stats = df.groupby(['ISO', 'Country']).agg({
        'Total Deaths': 'sum',
        'Total Affected': 'sum',
        'DisNo.': 'count',  # Number of disasters
        'Population': 'last'  # Latest population
    }).reset_index()
    
    country_stats = country_stats.rename(columns={'DisNo.': 'Disaster_Count'})
    
    # Calculate normalized metrics
    country_stats['Deaths_per_100k'] = (country_stats['Total Deaths'] / country_stats['Population']) * 100000
    country_stats['Affected_pct'] = (country_stats['Total Affected'] / country_stats['Population']) * 100
    country_stats['Disaster_Frequency'] = country_stats['Disaster_Count'] / 65  # Per year (1960-2024)
    
    # Create risk score (weighted combination)
    country_stats['Risk_Score'] = (
        country_stats['Deaths_per_100k'].rank(pct=True) * 0.4 +
        country_stats['Affected_pct'].rank(pct=True) * 0.3 +
        country_stats['Disaster_Frequency'].rank(pct=True) * 0.3
    ) * 100
    
    # Get top 10
    top_10 = country_stats.sort_values('Risk_Score', ascending=False).head(10)
    
    # Format for readability
    result = top_10[[
        'ISO', 'Country', 'Risk_Score', 'Disaster_Count', 
        'Total Deaths', 'Total Affected', 'Deaths_per_100k', 'Affected_pct'
    ]].round(2)
    
    # Save
    output_path = 'data/outputs/Top10_Vulnerable_Countries.xlsx'
    result.to_excel(output_path, index=False)
    
    print(f" Created: {output_path}")
    print("\n TOP 10 MOST VULNERABLE COUNTRIES:")
    print(result.to_string(index=False))
    
    return result

def create_disaster_type_impact():
    """Table 2: Impact by Disaster Type"""
    df = pd.read_csv('data/processed/master_disaster_table.csv')
    
    type_stats = df.groupby('Disaster Type').agg({
        'Total Deaths': ['sum', 'mean', 'max'],
        'Total Affected': ['sum', 'mean', 'max'],
        'DisNo.': 'count',
        'ISO': 'nunique'  # Countries affected
    }).round(2)
    
    # Flatten column names
    type_stats.columns = [
        'Total_Deaths', 'Avg_Deaths', 'Max_Deaths',
        'Total_Affected', 'Avg_Affected', 'Max_Affected',
        'Event_Count', 'Countries_Affected'
    ]
    
    # Add fatality rate
    type_stats['Fatality_Rate'] = (type_stats['Total_Deaths'] / type_stats['Total_Affected'] * 100).round(4)
    type_stats['Events_per_Year'] = (type_stats['Event_Count'] / 65).round(1)
    
    # Sort by total deaths
    type_stats = type_stats.sort_values('Total_Deaths', ascending=False)
    
    output_path = 'data/outputs/Disaster_Type_Impact_Analysis.xlsx'
    type_stats.to_excel(output_path)
    
    print(f"\n Created: {output_path}")
    print("\n DISASTER TYPE IMPACT ANALYSIS:")
    print(type_stats.head(10).to_string())
    
    return type_stats

def create_seasonal_patterns():
    """Table 3: Seasonal Disaster Patterns"""
    df = pd.read_csv('data/processed/features_engineered.csv')
    
    # Map months to seasons
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    df['Season'] = df['Month'].map(season_map)
    
    # Count events per season (use any column to count, like 'ISO')
    seasonal_stats = df.groupby('Season').agg({
        'ISO': 'count',  # Count events
        'Total Deaths': 'sum',
        'Total Affected': 'sum',
        'Priority': lambda x: (x == 'High').sum()
    }).rename(columns={
        'ISO': 'Events',
        'Priority': 'High_Priority_Events'
    })
    
    # Calculate percentages
    total_events = seasonal_stats['Events'].sum()
    seasonal_stats['Event_Pct'] = (seasonal_stats['Events'] / total_events * 100).round(1)
    seasonal_stats['High_Risk_Pct'] = (seasonal_stats['High_Priority_Events'] / seasonal_stats['Events'] * 100).round(1)
    
    # Add risk level
    conditions = [
        seasonal_stats['High_Risk_Pct'] >= 5,
        seasonal_stats['High_Risk_Pct'] >= 2,
        seasonal_stats['High_Risk_Pct'] < 2
    ]
    choices = ['Very High Risk', 'High Risk', 'Moderate Risk']
    seasonal_stats['Risk_Level'] = np.select(conditions, choices, default='Moderate Risk')
    
    seasonal_stats = seasonal_stats[[
        'Events', 'Event_Pct', 'High_Priority_Events', 'High_Risk_Pct',
        'Total Deaths', 'Total Affected', 'Risk_Level'
    ]]
    
    output_path = 'data/outputs/Seasonal_Disaster_Patterns.xlsx'
    seasonal_stats.to_excel(output_path)
    
    print(f"\n✅ Created: {output_path}")
    print("\n📅 SEASONAL DISASTER PATTERNS:")
    print(seasonal_stats.to_string())
    
    return seasonal_stats

def create_response_recommendations():
    """Table 4: Country-Specific Response Recommendations"""
    # Load multiple datasets
    country_risk = create_country_risk_ranking()
    disaster_impact = create_disaster_type_impact()
    
    # Load original for disaster type per country
    df = pd.read_csv('data/processed/master_disaster_table.csv')
    
    # Find most common disaster type per country
    country_disaster = df.groupby(['ISO', 'Disaster Type']).size().reset_index(name='count')
    idx = country_disaster.groupby('ISO')['count'].idxmax()
    primary_disaster = country_disaster.loc[idx][['ISO', 'Disaster Type']]
    primary_disaster = primary_disaster.rename(columns={'Disaster Type': 'Primary_Disaster_Type'})
    
    # Merge with risk data
    recommendations = pd.merge(
        country_risk,
        primary_disaster,
        on='ISO',
        how='left'
    )
    
    # Add response recommendations based on risk level and disaster type
    def generate_recommendation(row):
        if row['Risk_Score'] >= 80:
            priority = 'CRITICAL'
            prep = 'Pre-position supplies, establish emergency response teams'
        elif row['Risk_Score'] >= 60:
            priority = 'HIGH'
            prep = 'Regular drills, stockpile essential resources'
        else:
            priority = 'MEDIUM'
            prep = 'Monitoring, community awareness programs'
        
        disaster_specific = {
            'Flood': 'Early warning systems, flood barriers, evacuation plans',
            'Storm': 'Storm shelters, reinforced infrastructure, emergency communications',
            'Earthquake': 'Earthquake-resistant buildings, search & rescue teams, medical supplies',
            'Drought': 'Water conservation systems, food security programs, irrigation support'
        }
        
        disaster_prep = disaster_specific.get(row['Primary_Disaster_Type'], 'General disaster preparedness')
        
        return pd.Series({
            'Priority_Level': priority,
            'Recommended_Actions': f"{prep}. {disaster_prep}.",
            'Key_Metrics': f"{row['Disaster_Count']} historical events, {row['Deaths_per_100k']:.1f} deaths/100k"
        })
    
    # Apply recommendations
    rec_cols = recommendations.apply(generate_recommendation, axis=1)
    recommendations = pd.concat([recommendations, rec_cols], axis=1)
    
    # Select final columns
    final_table = recommendations[[
        'ISO', 'Country', 'Priority_Level', 'Risk_Score',
        'Primary_Disaster_Type', 'Disaster_Count', 'Deaths_per_100k',
        'Recommended_Actions', 'Key_Metrics'
    ]]
    
    output_path = 'data/outputs/Country_Response_Recommendations.xlsx'
    final_table.to_excel(output_path, index=False)
    
    print(f"\n Created: {output_path}")
    print("\n COUNTRY RESPONSE RECOMMENDATIONS:")
    print(final_table.to_string(index=False))
    
    return final_table

def create_all_tables():
    """Generate all detailed tables"""
    print("="*60)
    print("CREATING DETAILED ANALYTICAL TABLES")
    print("="*60)
    
    # Ensure output directory exists
    Path('data/outputs').mkdir(parents=True, exist_ok=True)
    
    # Create all tables
    print("\n1. Country Risk Ranking...")
    country_risk = create_country_risk_ranking()
    
    print("\n2. Disaster Type Impact Analysis...")
    disaster_impact = create_disaster_type_impact()
    
    print("\n3. Seasonal Patterns...")
    seasonal = create_seasonal_patterns()
    
    print("\n4. Response Recommendations...")
    recommendations = create_response_recommendations()
    
    # Create summary ZIP
    import zipfile
    with zipfile.ZipFile('data/outputs/Detailed_Analysis_Tables.zip', 'w') as zipf:
        for file in Path('data/outputs').glob('*.xlsx'):
            if 'RQ' not in file.name:  # Don't include RQ tables
                zipf.write(file, file.name)
    
    print("\n" + "="*60)
    print(" ALL TABLES CREATED SUCCESSFULLY")
    print("="*60)
    print("Outputs in data/outputs/:")
    print("1. Top10_Vulnerable_Countries.xlsx")
    print("2. Disaster_Type_Impact_Analysis.xlsx")
    print("3. Seasonal_Disaster_Patterns.xlsx")
    print("4. Country_Response_Recommendations.xlsx")
    print("5. Detailed_Analysis_Tables.zip (all tables)")
    print("="*60)

if __name__ == "__main__":
    create_all_tables()