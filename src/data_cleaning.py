import pandas as pd
import numpy as np
from pathlib import Path

def clean_emdat_data(input_path='data/processed/emdat_raw.csv'):
    """
    Clean EM-DAT disaster data
    """
    df = pd.read_csv(input_path)
    
    print(f" Cleaning EM-DAT data: {len(df)} rows")
    
    # 1. Filter to relevant columns
    keep_cols = [
        'DisNo.', 'ISO', 'Country', 'Region', 'Disaster Type', 'Disaster Subtype',
        'Start Year', 'Start Month', 'Start Day', 
        'Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Total Affected',
        'Total Damage', 'Latitude', 'Longitude'
    ]
    
    # Keep only columns that exist
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols]
    
    # 2. Handle missing ISO codes
    df['ISO'] = df['ISO'].fillna('Unknown')
    
    # 3. Convert numeric columns, handle non-numeric values
    numeric_cols = ['Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Total Affected', 'Total Damage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Create date column
    df['Start Date'] = pd.to_datetime(
        df[['Start Year', 'Start Month', 'Start Day']].fillna(1).astype(int).astype(str).agg('-'.join, axis=1),
        errors='coerce'
    )
    
    # 5. Filter to natural disasters only (remove technological)
    if 'Disaster Group' in df.columns:
        df = df[df['Disaster Group'] == 'Natural']
    
    print(f" EM-DAT cleaned: {len(df)} rows, {len(df.columns)} columns")
    return df

def clean_population_data(input_path='data/processed/population_raw.csv'):
    """
    Clean World Bank population data
    """
    df = pd.read_csv(input_path)
    
    print(f" Cleaning population data: {len(df)} rows")
    
    # 1. Remove regional aggregations (keep only country-level)
    # Regional codes: AFE, AFW, ARB, CSS, CEB, EAR, EAP, ECS, ECA, EMU, EUU, FCS, HIC, HPC, etc.
    regional_codes = ['AFE', 'AFW', 'ARB', 'CSS', 'CEB', 'EAR', 'EAP', 'ECS', 'ECA', 
                     'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 
                     'IDX', 'LAC', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MNA', 
                     'MIC', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 
                     'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS']
    
    df = df[~df['Country Code'].isin(regional_codes)]
    
    # 2. Convert from wide to long format (years as rows)
    id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    year_cols = [col for col in df.columns if col.isdigit() and 1960 <= int(col) <= 2024]
    
    df_long = pd.melt(
        df, 
        id_vars=id_vars,
        value_vars=year_cols,
        var_name='Year',
        value_name='Population'
    )
    
    # 3. Convert to numeric
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['Population'] = pd.to_numeric(df_long['Population'], errors='coerce')
    
    # 4. Drop rows with no population data
    df_long = df_long.dropna(subset=['Population'])
    
    print(f" Population data cleaned: {len(df_long)} rows")
    return df_long

def save_cleaned_data(emdat_df, pop_df):
    """Save cleaned datasets"""
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    emdat_df.to_csv('data/processed/emdat_cleaned.csv', index=False)
    pop_df.to_csv('data/processed/population_cleaned.csv', index=False)
    
    print(" Cleaned data saved to data/processed/")

if __name__ == "__main__":
    # Run cleaning pipeline
    emdat_clean = clean_emdat_data()
    pop_clean = clean_population_data()
    save_cleaned_data(emdat_clean, pop_clean)
    
    # Show summary
    print("\n CLEANING SUMMARY:")
    print(f"EM-DAT disasters: {len(emdat_clean):,} events")
    print(f"Population records: {len(pop_clean):,} country-year pairs")
    print(f"Countries in population data: {pop_clean['Country Code'].nunique()}")