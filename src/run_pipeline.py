# src/run_pipeline.py
"""
Main pipeline runner - executes all steps in sequence
"""

def run_ingestion():
    """Stage 1: Load raw data from source files"""
    print("\n" + "="*60)
    print("STAGE 1: DATA INGESTION")
    print("="*60)
    
    try:
        from data_ingestion import ingest_emdat_data, ingest_population_data
        
        print("📥 Loading EM-DAT disaster data...")
        emdat_df = ingest_emdat_data('data/raw/emdat_disasters.xlsx')
        
        print("📥 Loading World Bank population data...")
        pop_df = ingest_population_data('data/raw/population_wb.csv')
        
        # Save raw data
        emdat_df.to_csv('data/processed/emdat_raw.csv', index=False)
        pop_df.to_csv('data/processed/population_raw.csv', index=False)
        
        print(f"✅ Ingestion complete: {len(emdat_df):,} disasters, {len(pop_df):,} population records")
        return True
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False

def run_cleaning():
    """Stage 2: Clean and preprocess data"""
    print("\n" + "="*60)
    print("STAGE 2: DATA CLEANING")
    print("="*60)
    
    try:
        from data_cleaning import clean_emdat_data, clean_population_data, save_cleaned_data
        
        print("🧹 Cleaning disaster data...")
        emdat_clean = clean_emdat_data('data/processed/emdat_raw.csv')
        
        print("🧹 Cleaning population data...")
        pop_clean = clean_population_data('data/processed/population_raw.csv')
        
        save_cleaned_data(emdat_clean, pop_clean)
        
        print(f"✅ Cleaning complete: {len(emdat_clean):,} disasters, {len(pop_clean):,} population records")
        return True
        
    except Exception as e:
        print(f"❌ Cleaning failed: {e}")
        return False

def run_integration():
    """Stage 3: Merge disaster and population data"""
    print("\n" + "="*60)
    print("STAGE 3: DATA INTEGRATION")
    print("="*60)
    
    try:
        from data_integration import integrate_disaster_population, save_integrated_data
        
        print("🔗 Merging disaster and population data...")
        master_df, matched, unmatched = integrate_disaster_population(
            'data/processed/emdat_cleaned.csv',
            'data/processed/population_cleaned.csv'
        )
        
        save_integrated_data(master_df)
        
        print(f"✅ Integration complete: {len(master_df):,} total records")
        print(f"   Successfully matched: {matched:,} ({matched/len(master_df)*100:.1f}%)")
        print(f"   Unmatched: {unmatched:,} ({unmatched/len(master_df)*100:.1f}%)")
        return True
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False

def run_feature_engineering():
    """Stage 4: Create ML features"""
    print("\n" + "="*60)
    print("STAGE 4: FEATURE ENGINEERING")
    print("="*60)
    
    try:
        from feature_engineering import create_features, save_features
        
        print("⚙️ Creating features for ML models...")
        features_df = create_features('data/processed/master_disaster_table.csv')
        final_df = save_features(features_df)
        
        print(f"✅ Feature engineering complete: {len(final_df):,} samples, {len(final_df.columns)} features")
        print(f"   Priority distribution: {features_df['Priority'].value_counts().to_dict()}")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False

def run_model_training():
    """Stage 5: Train and evaluate models"""
    print("\n" + "="*60)
    print("STAGE 5: MODEL TRAINING")
    print("="*60)
    
    try:
        from model_training import load_and_prepare_data, traditional_threshold_method, train_ml_models, save_model_results
        
        print("🤖 Training ML models vs traditional methods...")
        X, y, le, df = load_and_prepare_data('data/processed/features_engineered.csv')
        trad_acc, df = traditional_threshold_method(df)
        results, X_test, y_test = train_ml_models(X, y, le)
        save_model_results(results)
        
        best_ml_acc = max([res['accuracy'] for res in results.values()])
        print(f"✅ Model training complete")
        print(f"   Traditional method: {trad_acc:.3f} accuracy")
        print(f"   Best ML model: {best_ml_acc:.3f} accuracy")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def run_disaster_specific_analysis():
    """Stage 6: Analyze disaster-type specific models"""
    print("\n" + "="*60)
    print("STAGE 6: DISASTER-SPECIFIC ANALYSIS")
    print("="*60)
    
    try:
        from disaster_specific_models import train_disaster_specific_models, compare_universal_vs_specific
        
        print("🌪️ Comparing universal vs disaster-specific models...")
        results, df = train_disaster_specific_models('data/processed/features_engineered.csv')
        universal_acc, weighted_acc = compare_universal_vs_specific(results, df)
        
        print(f"✅ Disaster-specific analysis complete")
        print(f"   Universal model: {universal_acc:.3f} accuracy")
        print(f"   Weighted specific models: {weighted_acc:.3f} accuracy")
        print(f"   Difference: {weighted_acc - universal_acc:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Disaster-specific analysis failed: {e}")
        return False

def main():
    """Main pipeline execution"""
    print("🚀 STARTING DISASTER RELIEF OPTIMIZATION PIPELINE")
    print("="*60)
    
    # Run all stages sequentially
    stages = [
        ("Data Ingestion", run_ingestion),
        ("Data Cleaning", run_cleaning),
        ("Data Integration", run_integration),
        ("Feature Engineering", run_feature_engineering),
        ("Model Training", run_model_training),
        ("Disaster-Specific Analysis", run_disaster_specific_analysis),
    ]
    
    successful_stages = 0
    
    for stage_name, stage_function in stages:
        print(f"\n▶️  Running: {stage_name}")
        if stage_function():
            successful_stages += 1
        else:
            print(f"⏹️  Pipeline stopped at: {stage_name}")
            break
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Stages completed: {successful_stages}/{len(stages)}")
    
    if successful_stages == len(stages):
        print("✅ ALL STAGES COMPLETED SUCCESSFULLY!")
        print("\n📁 Outputs available in:")
        print("   - data/processed/  (Intermediate data)")
        print("   - data/outputs/    (Final results)")
    else:
        print(f"⚠️  Pipeline incomplete. Failed at stage {successful_stages + 1}")
    
    print("="*60)

if __name__ == "__main__":
    main()