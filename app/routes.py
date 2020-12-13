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
import plotly.express as px
import plotly
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import base64
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


@app.route('/')
@app.route('/index')
def index():
    return redirect('/home')

@app.route('/home')
def test():
    data = {
        "user": "John Doe",
    }
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response

def is_map(text):
    arg_map = False

    if text.isdigit():
        arg_map = True
    else:
        arg_map = False

    return arg_map

def get_graph_jsn(text, is_map):

    centra = Centrality()
    node_path = centra.create_json_url(text, is_map)
    graph, jsn = centra.get_graph_url(node_path)

    return graph, jsn

def get_eigen_cent(graph):

    centra = Centrality()
    i_nodes = centra.get_eigen_centrality(graph)

    return i_nodes

@app.route('/eigen-cent-raw/<ids>', methods=["GET"])
def eigen_cent_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)


    data_list = []
    for tup in i_nodes:

        ID = tup[0]
        cent = tup[1]
        text = tup[2]
        data_dict = {"ID":ID, "centrality":cent, "text":text}
        data_list.append(data_dict)


    response = app.response_class(
        response=json.dumps(data_list),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/eigen-cent-vis-view/<ids>', methods=["GET"])
def eigen_cent_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)

    df = pd.DataFrame(i_nodes, columns=['id', 'cent', 'text'])
    df = df.sort_values('cent', ascending=False)

    df_sel = df.head(10)

    fig = px.bar(df_sel, x="text", y="cent", title="Eigenvector Centrality")
    fig.update_layout(
    autosize=False,
    width=500,
    height=500,
    xaxis=dict(
        showticklabels=False
        )
    )
    dv = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/eigen-cent-vis/<ids>', methods=["GET"])
def eigen_cent_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)

    df = pd.DataFrame(i_nodes, columns=['id', 'cent', 'text'])
    df = df.sort_values('cent', ascending=False)

    df_sel = df.head(10)

    fig = px.bar(df_sel, x="text", y="cent", title="Long-Form Input")
    fig.update_layout(
    autosize=False,
    width=500,
    height=500,
    xaxis=dict(
        showticklabels=False
        )
    )
    dv = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@app.route('/eigen-cent-cloud-vis/<ids>', methods=["GET"])
def eigen_cent_cloud_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)

    df = pd.DataFrame(i_nodes, columns=['id', 'cent', 'text'])
    df = df.sort_values('cent', ascending=False)
    df_sel = df.head(20)


    comment_words = ''
    stopwords = set(STOPWORDS)

    for val in df_sel['text']:
        val = str(val)
        tokens = val.split()

        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 200, height = 200,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)


    plt.figure(figsize = (2, 2), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')


    return render_template('plot.html', plot_url=plot_url)
