import pandas as pd
import os
from pathlib import Path

def ingest_emdat_data(file_path='data/raw/emdat_disasters.xlsx'):
    """
    Load EM-DAT Excel file into pandas DataFrame
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EM-DAT file not found at: {file_path}")
    
    # Load Excel file
    df = pd.read_excel(file_path)
    
    # Basic validation
    required_columns = ['ISO', 'Start Year', 'Disaster Type', 'Total Deaths']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from EM-DAT data")
    
    print(f" EM-DAT data loaded: {len(df)} rows, {len(df.columns)} columns")
    return df

def ingest_population_data(file_path='data/raw/Population_WB.csv'):
    """
    Load World Bank population CSV into pandas DataFrame
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Population file not found at: {file_path}")
    
    df = pd.read_csv(file_path, skiprows=4)  # WB CSVs have 4 header rows
    
    # Basic validation
    required_columns = ['Country Code', 'Country Name', '2020']  # Check recent year exists
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from population data")
    
    print(f"✅ Population data loaded: {len(df)} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    # Test the ingestion
    emdat_df = ingest_emdat_data()
    pop_df = ingest_population_data()
    
    # Save to processed folder for next step
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    emdat_df.to_csv('data/processed/emdat_raw.csv', index=False)
    pop_df.to_csv('data/processed/population_raw.csv', index=False)
    
    print(" Data ingestion complete. Files saved to data/processed/")