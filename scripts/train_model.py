# scripts/train_model.py - FIXED VERSION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

FEATURES = ['ST slope', 'exercise angina', 'chest pain type', 'max heart rate', 'oldpeak', 'sex', 'age']

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('data/cleveland1.csv')

    X = df[FEATURES]
    y = df['target']

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # FIX: Use set_output to preserve feature names
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=18, 
        min_samples_split=3, 
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n{'='*60}")
    print("Saving model and scaler...")
    joblib.dump(model, 'models/cardiac_disease_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    # Save feature names for reference
    with open('models/feature_names.txt', 'w') as f:
        f.write(','.join(FEATURES))

    print(f"✓ Saved: models/cardiac_disease_model.pkl")
    print(f"✓ Saved: models/scaler.pkl")
    print(f"✓ Saved: models/feature_names.txt")
    print(f"{'='*60}")
