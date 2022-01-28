from flask import Flask
from flask_restful import Api
import os

app = Flask(__name__)
api = Api(app)