import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle
import os

DATA_PATH = "data/titanic.csv"
MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(MODEL_DIR, "survival_model.pkl")

def get_preprocessed_data():
    df = pd.read_csv(DATA_PATH)
    
    # Feature selection
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].copy()
    y = df['Survived']
    
    # Simple imputation for Age and Fare
    imputer = SimpleImputer(strategy='median')
    X['Age'] = imputer.fit_transform(X[['Age']])
    X['Fare'] = imputer.fit_transform(X[['Fare']])
    
    # One-hot encoding for categorical variables (Sex and Embarked)
    X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Ensure all expected columns are present for the predictor tool
    expected_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
            
    # Reorder columns to ensure consistency
    X = X[expected_cols]
            
    return X, y

def train_model():
    print("Loading and preprocessing data...")
    X, y = get_preprocessed_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with validation accuracy: {accuracy:.4f}")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Saving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()
