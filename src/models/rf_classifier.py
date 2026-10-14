import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def train_rf_model():
    df = pd.read_csv("data/processed/virality_target_data.csv")
    
    features = ['category', 'time_period', 'duration_seconds', 'relatability_score', 'day_of_week']
    X = df[features].copy()
    y = df['is_viral']
    
    for col in ['category', 'time_period', 'day_of_week']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, random_state=42)
    rf.fit(X_train_bal, y_train_bal)
    
    y_pred = rf.predict(X_test)
    
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Non-Viral (0)", "Viral (1)"]))
    
    imp = rf.feature_importances_
    imp_df = pd.DataFrame({'Feature': features, 'Influence_Weight': imp})
    imp_df = imp_df.sort_values(by='Influence_Weight', ascending=False)
    imp_df['Influence_Weight'] = (imp_df['Influence_Weight'] * 100).round(2).astype(str) + "%"
    print(imp_df.to_string(index=False))

if __name__ == "__main__":
    train_rf_model()