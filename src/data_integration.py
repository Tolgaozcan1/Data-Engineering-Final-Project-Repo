
import pandas as pd
import numpy as np
from pathlib import Path

def integrate_disaster_population(emdat_path='data/processed/emdat_cleaned.csv',
                                 pop_path='data/processed/population_cleaned.csv'):
    """
    Merge disaster data with population data
    """
    # Load cleaned data
    disasters = pd.read_csv(emdat_path)
    population = pd.read_csv(pop_path)
    
    print(" Integrating disaster and population data...")
    print(f"Disasters: {len(disasters):,} events")
    print(f"Population records: {len(population):,}")
    
    # Ensure Year is integer in both datasets
    disasters['Start Year'] = disasters['Start Year'].astype(int)
    population['Year'] = population['Year'].astype(int)
    
    # Merge on ISO code and year
    merged = pd.merge(
        disasters,
        population,
        left_on=['ISO', 'Start Year'],
        right_on=['Country Code', 'Year'],
        how='left'  # Keep all disasters even if population data missing
    )
    
    # Calculate derived metrics
    merged['Deaths_per_100k'] = (merged['Total Deaths'] / merged['Population']) * 100000
    merged['Affected_pct'] = (merged['Total Affected'] / merged['Population']) * 100
    
    # Fill missing metrics with 0 (for disasters with no population match)
    merged['Deaths_per_100k'] = merged['Deaths_per_100k'].fillna(0)
    merged['Affected_pct'] = merged['Affected_pct'].fillna(0)
    
    # Count matches vs. non-matches
    matched = merged['Population'].notna().sum()
    unmatched = merged['Population'].isna().sum()
    
    print(f"\n Integration complete:")
    print(f"  Total merged records: {len(merged):,}")
    print(f"  Successfully matched: {matched:,} ({matched/len(merged)*100:.1f}%)")
    print(f"  Unmatched (no population data): {unmatched:,} ({unmatched/len(merged)*100:.1f}%)")
    
    return merged, matched, unmatched

def save_integrated_data(df, output_path='data/processed/master_disaster_table.csv'):
    """Save integrated master table"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Master table saved to: {output_path}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df):,}")

if __name__ == "__main__":
    # Run integration
    master_df, matched, unmatched = integrate_disaster_population()
    save_integrated_data(master_df)
    
    # Show sample of merged data
    print("\n SAMPLE OF MASTER TABLE:")
    print(master_df[['ISO', 'Country', 'Start Year', 'Disaster Type', 
                    'Total Deaths', 'Population', 'Deaths_per_100k']].head(10))