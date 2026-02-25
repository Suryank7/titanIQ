import pickle
import os
import pandas as pd
# from langchain_core.tools import tool
from langchain.tools import tool

MODEL_PATH = "backend/models/survival_model.pkl"

@tool
def predict_survival(pclass: int, sex: str, age: float, fare: float, embarked: str) -> str:
    """
    Predicts the survival probability of a passenger on the Titanic using an ML model.
    Use this when a user asks "what are my chances?", "what if...", or about specific passenger combinations.
    
    Arguments:
    - pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
    - sex: 'male' or 'female'
    - age: Age in years
    - fare: Ticket passenger fare
    - embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
    """
    try:
        if not os.path.exists(MODEL_PATH):
            return "Error: The survival prediction model has not been trained yet. Please run the training script."
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        # Create a DataFrame for the single prediction to match training features
        # Assuming training features: Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S
        # For simplicity, if we don't have SibSp/Parch, assume 0
        
        sex_male = 1 if sex.lower() == 'male' else 0
        embarked_q = 1 if embarked.upper() == 'Q' else 0
        embarked_s = 1 if embarked.upper() == 'S' else 0
        
        input_data = pd.DataFrame([{
            'Pclass': pclass,
            'Age': age,
            'SibSp': 0,
            'Parch': 0,
            'Fare': fare,
            'Sex_male': sex_male,
            'Embarked_Q': embarked_q,
            'Embarked_S': embarked_s
        }])
        
        probability = model.predict_proba(input_data)[0][1] # Probability of class 1 (Survived)
        survived = model.predict(input_data)[0]
        
        status = "survive" if survived == 1 else "not survive"
        
        return f"Prediction: The passenger would likely {status}. The calculated survival probability is {probability * 100:.2f}% based on the provided details."

    except Exception as e:
        return f"Error during prediction: {str(e)}"
