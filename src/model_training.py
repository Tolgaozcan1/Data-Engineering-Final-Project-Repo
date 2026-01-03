# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_prepare_data(filepath='data/processed/features_engineered.csv'):
    """Load and prepare data for modeling"""
    df = pd.read_csv(filepath)
    print(f" Loading data: {len(df):,} samples, {len(df.columns)} features")
    
    # Encode target variable
    le = LabelEncoder()
    df['Priority_encoded'] = le.fit_transform(df['Priority'])
    
    # Define features (X) and target (y)
    feature_cols = [
        'Month', 'Is_Flood', 'Is_Storm', 'Is_Earthquake', 'Is_Drought',
        'Has_Casualties', 'Has_Affected', 'Log_Population',
        'Country_Disaster_Count', 'Days_Since_Last'
    ]
    
    # Keep only columns that exist
    X_cols = [col for col in feature_cols if col in df.columns]
    X = df[X_cols]
    y = df['Priority_encoded']
    
    print(f"  Features: {len(X_cols)}")
    print(f"  Target classes: {len(le.classes_)} ({list(le.classes_)})")
    
    return X, y, le, df

def traditional_threshold_method(df):
    """
    Traditional method: Simple rule-based classification
    Compare with ML later
    """
    print("\n TRADITIONAL METHOD (Rule-based):")
    
    # Simple rules based on domain knowledge
    conditions = [
        (df['Deaths_per_100k'] >= 10) | (df['Total Deaths'] >= 1000),
        (df['Deaths_per_100k'] >= 1) | (df['Total Deaths'] >= 100),
        (df['Deaths_per_100k'] < 1) & (df['Total Deaths'] < 100)
    ]
    choices = ['High', 'Medium', 'Low']
    df['Traditional_Priority'] = np.select(conditions, choices, default='Low')
    
    # Encode for comparison
    le = LabelEncoder()
    y_true = le.fit_transform(df['Priority'])
    y_pred = le.transform(df['Traditional_Priority'])
    
    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy: {acc:.3f}")
    print("  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    return acc, df

def train_ml_models(X, y, le):
    """
    Train and evaluate ML models
    """
    print("\n MACHINE LEARNING MODELS:")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE or resampling
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    
    # Models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  {name}:")
        
        # Train
        model.fit(X_train_bal, y_train_bal)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=5)
        
        print(f"    Test Accuracy: {acc:.3f}")
        print(f"    CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        print(f"    Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Feature importance (for Random Forest)
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"    Top 5 features:")
            for _, row in feature_importance.head().iterrows():
                print(f"      {row['feature']}: {row['importance']:.3f}")
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'cv_score': cv_scores.mean(),
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    return results, X_test, y_test

def save_model_results(results, output_dir='data/outputs'):
    """Save model results and figures"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    comparison = []
    for name, res in results.items():
        comparison.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'CV_Score': res['cv_score']
        })
    
    df_compare = pd.DataFrame(comparison)
    df_compare.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    
    # Plot confusion matrix for best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_result = results[best_model_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'], ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    axes[1].bar(models, accuracies, color=['skyblue', 'lightcoral'])
    axes[1].set_title('Model Accuracy Comparison')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n Results saved to {output_dir}/")
    print(f"  - model_comparison.csv")
    print(f"  - model_results.png")
    
    return df_compare

if __name__ == "__main__":
    print("="*60)
    print("MODEL TRAINING: ML vs Traditional Methods (RQ2)")
    print("="*60)
    
    # 1. Load data
    X, y, le, df = load_and_prepare_data()
    
    # 2. Traditional method baseline
    trad_acc, df = traditional_threshold_method(df)
    
    # 3. ML models
    results, X_test, y_test = train_ml_models(X, y, le)
    
    # 4. Save results
    df_compare = save_model_results(results)
    
    # 5. Answer RQ2
    print("\n" + "="*60)
    print(" RQ2 ANSWER: ML vs Traditional Methods")
    print("="*60)
    
    best_ml_acc = max([res['accuracy'] for res in results.values()])
    improvement = best_ml_acc - trad_acc
    
    print(f"Traditional Method Accuracy: {trad_acc:.3f}")
    print(f"Best ML Model Accuracy:     {best_ml_acc:.3f}")
    print(f"Improvement:                {improvement:.3f} ({improvement/trad_acc*100:.1f}%)")
    
    if improvement > 0:
        print(" CONCLUSION: ML models outperform traditional rule-based methods")
    else:
        print(" CONCLUSION: Traditional methods competitive with ML")
    
    print("="*60)