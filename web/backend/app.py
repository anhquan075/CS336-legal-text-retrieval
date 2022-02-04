from flask import Flask
from flask_restful import Api

from transformers import pipeline

MODEL_NAME = "hogger32/xlmRoberta-for-VietnameseQA"

# Initialize the flask instance
app = Flask(__name__)
api = Api(app)

# Initialize QA model
nlp = pipeline('question-answering', model=MODEL_NAME, tokenizer=MODEL_NAME)