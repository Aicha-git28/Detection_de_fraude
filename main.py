from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Charger le modèle RandomForest sauvegardé
model = joblib.load("modele_random_forest_final.pkl")

# Créer l'app FastAPI
app = FastAPI(
    title="API Détection de Fraude",
    description="Prédit si une transaction est frauduleuse et retourne la probabilité."
)

# Définir le format des données attendues
class TransactionData(BaseModel):
    features: list  # liste des features numériques ex : [39.79, 93213.17, 7, ...]

# Route de test pour vérifier que l'API tourne
@app.get("/")
def read_root():
    return {"message": "API de détection de fraude prête "}

# Route POST pour faire une prédiction
@app.post("/predict")
def predict(data: TransactionData):
    X = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    probability = round(model.predict_proba(X)[0,1], 4)
    return {
        "prediction": prediction,
        "probability_fraud": probability
    }
