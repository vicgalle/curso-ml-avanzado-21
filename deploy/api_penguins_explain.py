
from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel
import pandas as pd
import numpy as np

import lime
import lime.lime_tabular


# cargamos el modelo
pipe = load('models/model_2021-07-22*17-59-05.joblib') 

# Cargamos el explicador
df = pd.read_csv('data/penguins.csv')
df = df[['Species', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']]
y = df['Species']
X = df.drop('Species', axis=1)
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)
explainer = lime.lime_tabular.LimeTabularExplainer(X_tr.values, feature_names=df.columns[1:].values, class_names=df['Species'].unique(), discretize_continuous=False)


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


@app.post("/explain")
def explain(params: ModelFeatures):
    x = np.asarray([[params.culmen_length, params.culmen_depth, params.flipper_length, params.body_mass]]).squeeze(0)
    print(x.shape)
    exp = explainer.explain_instance(x, pipe.predict_proba, num_features=4, top_labels=1)
    return {'explanation' : str(exp.as_map())}
