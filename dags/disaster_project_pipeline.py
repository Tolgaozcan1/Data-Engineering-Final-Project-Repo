from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

# Add project root to path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def ingest_data():
    """Task 1: Data Ingestion"""
    from src.data_ingestion import ingest_emdat_data, ingest_population_data
    import pandas as pd
    
    print("📥 Starting data ingestion...")
    emdat_df = ingest_emdat_data('data/raw/emdat_disasters.xlsx')
    pop_df = ingest_population_data('data/raw/population_wb.csv')
    
    # Save raw data
    import pandas as pd
    emdat_df.to_csv('data/processed/emdat_raw.csv', index=False)
    pop_df.to_csv('data/processed/population_raw.csv', index=False)
    
    print("Data ingestion complete")

def clean_data():
    """Task 2: Data Cleaning"""
    from src.data_cleaning import clean_emdat_data, clean_population_data, save_cleaned_data
    
    print("🧹 Starting data cleaning...")
    emdat_clean, _, _ = clean_emdat_data('data/processed/emdat_raw.csv')
    pop_clean, _, _, _, _ = clean_population_data('data/processed/population_raw.csv')
    save_cleaned_data(emdat_clean, pop_clean)
    
    print(" Data cleaning complete")

def integrate_data():
    """Task 3: Data Integration"""
    from src.data_integration import integrate_disaster_population, save_integrated_data
    
    print("🔗 Starting data integration...")
    master_df, matched, unmatched = integrate_disaster_population(
        'data/processed/emdat_cleaned.csv',
        'data/processed/population_cleaned.csv'
    )
    save_integrated_data(master_df)
    
    print(" Data integration complete")

def engineer_features():
    """Task 4: Feature Engineering"""
    from src.feature_engineering import create_features, save_features
    
    print("⚙️ Starting feature engineering...")
    features_df = create_features('data/processed/master_disaster_table.csv')
    save_features(features_df)
    
    print(" Feature engineering complete")

def train_models():
    """Task 5: Model Training"""
    from src.model_training import load_and_prepare_data, traditional_threshold_method, train_ml_models, save_model_results
    
    print("🤖 Starting model training...")
    X, y, le, df = load_and_prepare_data('data/processed/features_engineered.csv')
    trad_acc, df = traditional_threshold_method(df)
    results, X_test, y_test = train_ml_models(X, y, le)
    save_model_results(results)
    
    print(" Model training complete")

def disaster_specific_analysis():
    """Task 6: Disaster-Specific Analysis"""
    from src.disaster_specific_models import train_disaster_specific_models, compare_universal_vs_specific
    
    print("🌪️ Starting disaster-specific analysis...")
    results, df = train_disaster_specific_models('data/processed/features_engineered.csv')
    universal_acc, weighted_acc = compare_universal_vs_specific(results, df)
    
    print(" Disaster-specific analysis complete")

def generate_report():
    """Task 7: Generate Final Report"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print(" Generating final report...")
    
    # Create summary statistics
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_disasters': pd.read_csv('data/processed/features_engineered.csv').shape[0],
        'countries_covered': pd.read_csv('data/processed/features_engineered.csv')['ISO'].nunique(),
        'time_period': f"{pd.read_csv('data/processed/features_engineered.csv')['Start Year'].min()}-{pd.read_csv('data/processed/features_engineered.csv')['Start Year'].max()}"
    }
    
    # Save summary
    import json
    with open('data/outputs/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(" Final report generated")
    print(f"Summary: {summary}")

# Define DAG
default_args = {
    'owner': 'disaster_relief_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'disaster_relief_pipeline',
    default_args=default_args,
    description='End-to-end pipeline for disaster relief optimization',
    schedule_interval='@monthly',  # Run monthly for updated data
    catchup=False,
    tags=['disaster', 'optimization', 'ml'],
)

# Define tasks
start_task = DummyOperator(task_id='start_pipeline', dag=dag)
end_task = DummyOperator(task_id='end_pipeline', dag=dag)

ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag,
)

integrate_task = PythonOperator(
    task_id='integrate_data',
    python_callable=integrate_data,
    dag=dag,
)

features_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

specific_task = PythonOperator(
    task_id='disaster_specific_analysis',
    python_callable=disaster_specific_analysis,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    dag=dag,
)

# Define workflow
start_task >> ingest_task >> clean_task >> integrate_task >> features_task >> train_task >> specific_task >> report_task >> end_task