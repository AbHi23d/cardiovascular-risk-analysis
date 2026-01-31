import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier # Removed due to libomp missing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
def load_and_preprocess(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Basic stats
    print(f"Shape: {df.shape}")
    print(df.head())

    # Encode binary variables
    binary_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() == 2]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col}: {le.classes_}")

    # One-hot encode multi-class variables
    # Identify remaining object columns
    multi_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    print(f"Shape after encoding: {df.shape}")
    
    return df

def train_and_evaluate(df):
    X = df.drop('HeartDisease', axis=1)
    # Important: HeartDisease was encoded. If 'Yes' was 1, we are good.
    # Usually LabelEncoder maps 'No'->0, 'Yes'->1 alphabetically, which is correct.
    y = df['HeartDisease']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        # Gradient Boosting doesn't support class_weight param directly in sklearn
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} AUC: {auc}")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        
    return results

if __name__ == "__main__":
    filepath = "data/heart_2020_cleaned.csv"
    df = load_and_preprocess(filepath)
    results = train_and_evaluate(df)
