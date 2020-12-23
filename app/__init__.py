from flask import Flask
import os

application = Flask(__name__)
application.secret_key = os.urandom(16)

from app import routes
