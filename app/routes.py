from flask import render_template, request, redirect, session, Markup
from . import app
import pandas as pd
from urllib.request import urlopen
from app.centrality import Centrality
import requests
import json
import urllib
import tempfile
import os
import uuid


@app.route('/')
@app.route('/index')
def index():
    return redirect('/home')

@app.route('/home')
def my_form():
    return render_template('index.html')
