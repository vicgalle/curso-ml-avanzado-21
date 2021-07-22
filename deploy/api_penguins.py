
from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel


# cargamos el modelo
pipe = load('models/model_2021-07-22*17-59-05.joblib') 

def get_prediction(params):
    
    x = [[params.culmen_length, params.culmen_depth, params.flipper_length, params.body_mass]]
    y = pipe.predict(x)[0]  # just get single value
    prob = pipe.predict_proba(x)[0].tolist()  # send to list for return
    return {'prediction': y, 'probability': prob}


# initiate API
app = FastAPI()

# Definimos una clase anotando los tipos de las features
class ModelFeatures(BaseModel):
    culmen_length: float
    culmen_depth: float
    flipper_length: float
    body_mass: float


@app.post("/predict")
def predict(params: ModelFeatures):
    prediction = get_prediction(params)
    return prediction
