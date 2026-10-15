'''
FastAPI app for the network security project.
  /         -> redirects to the Swagger docs page
  /train    -> runs the full training pipeline
  /predict  -> upload a CSV, get predictions back as an HTML table
Run with: uvicorn app:app --reload
'''

import sys
import os

import certifi
ca = certifi.where()  # CA certificates for a secure MongoDB connection

from dotenv import load_dotenv
load_dotenv()  # load MONGO_DB_URL and other secrets from .env
mongo_db_url = os.getenv("MONGO_DB_URL")  # same name as in .env (course used MONGODB_URL_KEY)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# MongoDB connection (the training pipeline also opens its own in data_ingestion.py)
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()

# CORS: lets a web page on another domain call this API. "*" allows everyone, fine for learning
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")  # HTML templates, e.g. templates/table.html


@app.get("/", tags=["authentication"])
async def index():
    '''Home page just sends you to the Swagger UI.'''
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    '''Run the whole training pipeline (ingestion to model trainer).'''
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    '''Read an uploaded CSV, predict each row, save output.csv and show results as a table.'''
    try:
        df = pd.read_csv(file.file)

        # saved during training: fitted preprocessor + best model
        preprocesor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor, model=final_model)  # transform + predict in one call

        print(df.iloc[0])  # debug: first input row
        y_pred = network_model.predict(df)  # 1 = legitimate, 0 = phishing
        print(y_pred)

        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        # df['predicted_column'].replace(-1, 0)
        # return df.to_json()

        df.to_csv('prediction_output/output.csv')  # folder must already exist
        table_html = df.to_html(classes='table table-striped')  # dataframe -> HTML table

        # request is passed separately, not inside the context dict
        return templates.TemplateResponse(request=request, name="table.html", context={"table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    # lets you run "python app.py" instead of uvicorn; 0.0.0.0 makes it reachable from other machines/containers
    app_run(app, host="0.0.0.0", port=8000)