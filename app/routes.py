from flask import render_template, request, redirect, session, Markup, send_file
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
import plotly.graph_objects as go
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

def load_nodesets_for_corpus(corpus_name):
    directory = 'http://corpora.aifdb.org/' + corpus_name + '/nodesets'
    with urlopen(directory) as url:
        data = json.load(url)
    return data

def load_from_cache(aifdb_id):
    directory = './cache/'
    filename = str(aifdb_id) + '.json'
    full_file_name = directory + filename
    data = ''
    file_found = False
    try:
        with open(full_file_name) as json_file:
            data = json.load(json_file)
        file_found = True
    except:
        file_found = False
    return file_found, data

def save_to_cache(aifdb_id, jsn_data):
    directory = './cache/'
    filename = str(aifdb_id) + '.json'
    full_file_name = directory + filename

    with open(full_file_name,"w") as fo:
        json.dump(jsn_data, fo)

def load_nodesets_from_cache(aifdb_id):
    directory = './cache/'
    filename = str(aifdb_id) + '_nodesets.json'
    full_file_name = directory + filename
    data = ''
    file_found = False
    try:
        with open(full_file_name) as json_file:
            data = json.load(json_file)
        file_found = True
    except:
        file_found = False
    return file_found, data

def save_nodesets_to_cache(aifdb_id, jsn_data):
    directory = './cache/'
    filename = str(aifdb_id) + '_nodesets.json'
    full_file_name = directory + filename

    with open(full_file_name,"w") as fo:
        json.dump(jsn_data, fo)

def get_graph_jsn(text, is_map):

    file_found = False
    jsn = ''
    if not is_map:
        nodeset_file_found,nodeset_list = load_nodesets_from_cache(text)
        if nodeset_file_found:
            nodesets = load_nodesets_for_corpus(text)
            if nodeset_list == nodesets:
                file_found, jsn = load_from_cache(text)
            else:
                file_found = False
        else:
            file_found = False
    else:
        file_found, jsn = load_from_cache(text)



    centra = Centrality()
    graph = ''
    if not file_found:

        node_path = centra.create_json_url(text, is_map)
        graph, jsn = centra.get_graph_url(node_path)
        save_to_cache(text, jsn)

        if not is_map:
            nodesets = load_nodesets_for_corpus(text)
            save_nodesets_to_cache(text, nodesets)
    else:

        graph = centra.get_graph_string(jsn)
    if not is_map:
        graph = centra.remove_an_nodes(graph)
        graph = centra.remove_iso_nodes(graph)

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

    dv = make_bar_chart(df_sel, "text", "cent", "Eigenvector Centrality", None, None)
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

    dv = make_bar_chart(df_sel, "text", "cent", "Eigenvector Centrality", 500, 500)


    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

def make_bar_chart(dataframe, x_axis, y_axis, title, width, height):
    fig = px.bar(dataframe, x=x_axis, y=y_axis, title=title)
    fig.update_layout(
    autosize=True,
    width=width,
    height=height,
    xaxis=dict(
        showticklabels=False
        ),
    yaxis=dict(automargin=True)
    )
    dv = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)

    return dv

def make_gauge_chart(value, title, min_value, max_value):

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {'axis': {'range': [min_value, max_value]}}))

    dv = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
    return dv

def make_line_chart(dataframe, x, y, colour, line_group, hover_name, width, height):

    fig = px.line(dataframe, x=x, y=y, color=colour,
              line_group=line_group, hover_name=hover_name)
    fig.update_layout(
    autosize=True,
    width=width,
    height=height
    )

    dv = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
    return dv

def make_sunburst_chart(dataframe, path, values, title,width, height):

    fig = px.sunburst(dataframe,
                  path=path,
                  values=values,
                  title=title,
                  width=width, height=height)

    dv = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
    return dv

@app.route('/eigen-cent-cloud-vis-view/<ids>', methods=["GET"])
def eigen_cent_cloud_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)

    df = pd.DataFrame(i_nodes, columns=['id', 'cent', 'text'])
    df = df.sort_values('cent', ascending=False)
    df_sel = df.head(20)


    plot_data = make_word_cloud(df_sel, "text", 500, 500)


    return render_template('plot.html', plot_url=plot_data)

@app.route('/eigen-cent-cloud-vis/<ids>', methods=["GET"])
def eigen_cent_cloud_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)

    df = pd.DataFrame(i_nodes, columns=['id', 'cent', 'text'])
    df = df.sort_values('cent', ascending=False)
    df_sel = df.head(20)

    plot_data = make_word_cloud(df_sel, "text", 500, 500)


    response = app.response_class(
        response=json.dumps(plot_data),
        status=200,
        mimetype='application/json'
    )
    return response

def make_word_cloud(dataframe, text_column, width, height):
    comment_words = ''
    stopwords = set(STOPWORDS)

    for val in dataframe[text_column]:
        val = str(val)
        tokens = val.split()

        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = width, height = height,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 8).generate(comment_words)

    fig_x = width / 100
    fig_y = height / 100

    plt.figure(figsize = (fig_x, fig_y), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    plot_data = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_data


def get_cogency(graph, centra):
    yas = centra.get_ass_ya(graph)
    i_nodes_yas = centra.get_ya_i_nodes(graph, yas)
    ra_list, ca_list = centra.get_i_ra_ca_nodes(graph, i_nodes_yas)

    cogency = len(ra_list) / len(yas)

    print(len(ra_list), len(yas))
    return cogency
def get_cogency_ca(graph, centra):
    yas = centra.get_ass_ya(graph)
    i_nodes_yas = centra.get_ya_i_nodes(graph, yas)
    ra_list, ca_list = centra.get_i_ra_ca_nodes(graph, i_nodes_yas)

    cogency = len(ca_list) / len(yas)


    return cogency

@app.route('/cogency-raw/<ids>', methods=["GET"])
def cogency_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    cogency = get_cogency(graph, centra)

    response = app.response_class(
        response=json.dumps(cogency),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/cogency-vis/<ids>', methods=["GET"])
def cogency_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    cogency = get_cogency(graph, centra)

    dv = make_gauge_chart(cogency, 'Cogency', None, 1)

    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@app.route('/cogency-vis-view/<ids>', methods=["GET"])
def cogency_vis_view(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    cogency = get_cogency(graph, centra)

    dv = make_gauge_chart(cogency, 'Cogency', None, 1)

    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

def get_correctness(graph, centra):

    l_nodes = centra.get_l_node_list(graph)
    ta_nodes = centra.get_l_ta_nodes(graph, l_nodes)
    correctness = len(ta_nodes) / len(l_nodes)
    return correctness

@app.route('/correctness-raw/<ids>', methods=["GET"])
def correctness_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    correctness = get_correctness(graph, centra)

    response = app.response_class(
        response=json.dumps(correctness),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/correctness-vis/<ids>', methods=["GET"])
def correctness_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    correctness = get_correctness(graph, centra)
    dv = make_gauge_chart(correctness, 'Correctness', None, 1)

    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response
@app.route('/correctness-vis-view/<ids>', methods=["GET"])
def correctness_vis_view(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    correctness = get_correctness(graph, centra)
    dv = make_gauge_chart(correctness, 'Correctness', None, 1)
    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

def get_coherence(graph, centra):
    graph = centra.remove_redundant_nodes(graph)
    isos = centra.get_isolated_nodes(graph)
    coherence = 1/len(isos)
    return coherence

@app.route('/coherence-raw/<ids>', methods=["GET"])
def coherence_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    coherence = get_coherence(graph, centra)

    response = app.response_class(
        response=json.dumps(coherence),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/coherence-vis/<ids>', methods=["GET"])
def coherence_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    coherence = get_coherence(graph, centra)

    dv = make_gauge_chart(coherence, 'Coherence', None, 1)

    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response
@app.route('/coherence-vis-view/<ids>', methods=["GET"])
def coherence_vis_view(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    coherence = get_coherence(graph, centra)

    dv = make_gauge_chart(coherence, 'Coherence', None, 1)

    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

def get_popularity(graph, centra):
    i_nodes = centra.get_i_node_ids(graph)
    yas, i_node_tups = centra.get_i_ya_nodes(graph, i_nodes)
    return i_node_tups

def get_unpopularity(graph, centra):
    i_nodes = centra.get_i_node_ids(graph)
    yas, i_node_tups = centra.get_i_ya_dis_nodes(graph, i_nodes)
    return i_node_tups


@app.route('/unpopularity-raw/<ids>', methods=["GET"])
def unpopularity_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)

    response = app.response_class(
        response=json.dumps(popularity_list),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/unpopularity-vis-view/<ids>', methods=["GET"])
def unpopularity_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)

    df = pd.DataFrame(popularity_list, columns=['UnPopularity','Text'])
    df = df.sort_values('UnPopularity', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "UnPopularity", "UnPopularity Top 10", 800, 500)
    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/unpopularity-vis/<ids>', methods=["GET"])
def unpopularity_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)

    df = pd.DataFrame(popularity_list, columns=['UnPopularity','Text'])
    df = df.sort_values('UnPopularity', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "UnPopularity", "UnPopularity Top 10", 800, 500)


    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response



@app.route('/popularity-raw/<ids>', methods=["GET"])
def popularity_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)

    response = app.response_class(
        response=json.dumps(popularity_list),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/popularity-vis-view/<ids>', methods=["GET"])
def popularity_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)

    df = pd.DataFrame(popularity_list, columns=['Popularity','Text'])
    df = df.sort_values('Popularity', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "Popularity", "Popularity Top 10", 800, 500)
    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/popularity-vis/<ids>', methods=["GET"])
def popularity_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)

    df = pd.DataFrame(popularity_list, columns=['Popularity','Text'])
    df = df.sort_values('Popularity', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "Popularity", "Popularity Top 10", 800, 500)


    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

def get_appeal(graph, centra, popularity_list):

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    popularity_2_list = centra.get_ra_ma_speaker_count(graph, new_i_nodes,centra)
    df1 = pd.DataFrame(popularity_list, columns =['Val', 'Text'])
    df2 = pd.DataFrame(popularity_2_list, columns =['Val', 'Text'])

    df1 = df1.set_index(['Text'])
    df2 = df2.set_index(['Text'])

    merge_df = df1.add(df2, fill_value=0)

    appeal_list = merge_df.to_records().tolist()
    return appeal_list

@app.route('/appeal-vis-view/<ids>', methods=["GET"])
def appeal_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)
    appeal_list = get_appeal(graph, centra, popularity_list)

    df = pd.DataFrame(appeal_list, columns=['Text', 'Appeal'])
    df = df.sort_values('Appeal', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "Appeal", "Appeal Top 10", 800, 500)
    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/appeal-vis/<ids>', methods=["GET"])
def appeal_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)
    appeal_list = get_appeal(graph, centra, popularity_list)

    df = pd.DataFrame(appeal_list, columns=['Text', 'Appeal'])
    df = df.sort_values('Appeal', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "Appeal", "Appeal Top 10", 800, 500)


    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

def get_unappeal(graph, centra, popularity_list):

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    popularity_2_list = centra.get_ca_ma_speaker_count(graph, new_i_nodes,centra)
    df1 = pd.DataFrame(popularity_list, columns =['Val', 'Text'])
    df2 = pd.DataFrame(popularity_2_list, columns =['Val', 'Text'])

    df1 = df1.set_index(['Text'])
    df2 = df2.set_index(['Text'])

    merge_df = df1.add(df2, fill_value=0)

    appeal_list = merge_df.to_records().tolist()
    return appeal_list



@app.route('/appeal-raw/<ids>', methods=["GET"])
def appeal_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)
    appeal_list = get_appeal(graph, centra, popularity_list)

    response = app.response_class(
        response=json.dumps(appeal_list),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/unappeal-raw/<ids>', methods=["GET"])
def unappeal_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)
    appeal_list = get_unappeal(graph, centra, popularity_list)

    response = app.response_class(
        response=json.dumps(appeal_list),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/unappeal-vis-view/<ids>', methods=["GET"])
def unappeal_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)
    appeal_list = get_unappeal(graph, centra, popularity_list)

    df = pd.DataFrame(appeal_list, columns=['Text', 'UnAppeal'])
    df = df.sort_values('UnAppeal', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "UnAppeal", "UnAppeal Top 10", 500, 500)
    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/unappeal-vis/<ids>', methods=["GET"])
def unappeal_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)
    appeal_list = get_unappeal(graph, centra, popularity_list)

    df = pd.DataFrame(appeal_list, columns=['Text', 'UnAppeal'])
    df = df.sort_values('UnAppeal', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "UnAppeal", "UnAppeal Top 10", 500, 500)


    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

def get_node_divisiveness(ra_list,ca_list):
    ra_count = len(ra_list)
    div_scores = []
    for ca_tup in ca_list:
        ca_id = ca_tup[0]
        ra_ca_count = len(ca_tup[1])

        temp_div = ra_count + ra_ca_count
        div_scores.append(temp_div)
    return div_scores
def get_divisiveness(graph, centra, i_nodes):
    node_div = []
    for i in i_nodes:
        node_id = i[0]
        text = i[1]
        ra_list, ca_list, ca_ra_list = centra.get_i_ca_nodes(graph, centra, node_id)
        div_list = get_node_divisiveness(ra_list, ca_ra_list)
        div = sum(div_list)
        i_node_div_tup = (node_id, text, div)
        node_div.append(i_node_div_tup)
    return node_div

@app.route('/divisiveness-raw/<ids>', methods=["GET"])
def divisiveness_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    divisiveness_list = get_divisiveness(graph, centra, i_nodes)

    response = app.response_class(
        response=json.dumps(divisiveness_list),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/divisiveness-vis-view/<ids>', methods=["GET"])
def divisiveness_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    divisiveness_list = get_divisiveness(graph, centra, i_nodes)

    df = pd.DataFrame(divisiveness_list, columns=['ID','Text', 'Divisiveness'])
    df = df.sort_values('Divisiveness', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "Divisiveness", "Divisiveness Top 10", None, None)
    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/divisiveness-vis/<ids>', methods=["GET"])
def divisiveness_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    divisiveness_list = get_divisiveness(graph, centra, i_nodes)

    df = pd.DataFrame(divisiveness_list, columns=['ID','Text', 'Divisiveness'])
    df = df.sort_values('Divisiveness', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "Divisiveness", "Divisiveness Top 10", None, None)


    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@app.route('/s-node-timeline-raw/<ids>', methods=["GET"])
def s_node_timeline_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_nodes = centra.get_l_node_list(graph)
    ta_timeline = centra.get_l_ta_s_nodes(graph, l_nodes)

    df = pd.DataFrame(ta_timeline, columns =['TA Count', 'Node Type', 'Node Count'])

    timeline_list = df.to_dict(orient='records')

    response = app.response_class(
        response=json.dumps(timeline_list),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/s-node-timeline-vis/<ids>', methods=["GET"])
def s_node_timeline_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_nodes = centra.get_l_node_list(graph)
    ta_timeline = centra.get_l_ta_s_nodes(graph, l_nodes)

    df = pd.DataFrame(ta_timeline, columns =['TA Count', 'Node Type', 'Node Count'])

    dv = make_line_chart(df, 'TA Count', 'Node Count', 'Node Type', 'Node Type', 'Node Type', 600, 600)

    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@app.route('/s-node-timeline-vis-view/<ids>', methods=["GET"])
def s_node_timeline_vis_view(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_nodes = centra.get_l_node_list(graph)
    ta_timeline = centra.get_l_ta_s_nodes(graph, l_nodes)

    df = pd.DataFrame(ta_timeline, columns =['TA Count', 'Node Type', 'Node Count'])

    dv = make_line_chart(df, 'TA Count', 'Node Count', 'Node Type', 'Node Type', 'Node Type', 600, 300)

    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/statistics-raw/<ids>', methods=["GET"])
def statistics_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)

    data = jsn['nodes']
    df_nodes = pd.DataFrame.from_dict(data, orient='columns')

    RA_nodes = df_nodes[df_nodes['type']=='RA']
    CA_nodes = df_nodes[df_nodes['type']=='CA']
    MA_nodes = df_nodes[df_nodes['type']=='MA']
    YA_nodes = df_nodes[df_nodes['type']=='YA']

    ras = RA_nodes['text'].value_counts().to_frame().reset_index()
    ras.columns = ['text', 'count']
    ras['type']='RA'

    cas = CA_nodes['text'].value_counts().to_frame().reset_index()
    cas.columns = ['text', 'count']
    cas['type']='CA'

    mas = MA_nodes['text'].value_counts().to_frame().reset_index()
    mas.columns = ['text', 'count']
    mas['type']='MA'

    yas = YA_nodes['text'].value_counts().to_frame().reset_index()
    yas.columns = ['text', 'count']
    yas['type']='YA'

    overall_df = pd.concat([ras, cas, mas, yas], ignore_index=True)

    l_node_count = len(l_nodes)
    i_node_count = len(i_nodes)

    new_loc_row = {'text':'Locutions', 'count':l_node_count, 'type':'Locution'}
    new_prop_row = {'text':'Propositions', 'count':i_node_count, 'type':'I-node'}

    overall_df = overall_df.append(new_loc_row, ignore_index=True)
    overall_df = overall_df.append(new_prop_row, ignore_index=True)

    data_dict = overall_df.to_dict(orient='records')

    response = app.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/statistics-vis-view/<ids>', methods=["GET"])
def statistics_vis_view(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)

    data = jsn['nodes']
    df_nodes = pd.DataFrame.from_dict(data, orient='columns')

    RA_nodes = df_nodes[df_nodes['type']=='RA']
    CA_nodes = df_nodes[df_nodes['type']=='CA']
    MA_nodes = df_nodes[df_nodes['type']=='MA']
    YA_nodes = df_nodes[df_nodes['type']=='YA']

    ras = RA_nodes['text'].value_counts().to_frame().reset_index()
    ras.columns = ['text', 'count']
    ras['type']='RA'

    cas = CA_nodes['text'].value_counts().to_frame().reset_index()
    cas.columns = ['text', 'count']
    cas['type']='CA'

    mas = MA_nodes['text'].value_counts().to_frame().reset_index()
    mas.columns = ['text', 'count']
    mas['type']='MA'

    yas = YA_nodes['text'].value_counts().to_frame().reset_index()
    yas.columns = ['text', 'count']
    yas['type']='YA'

    overall_df = pd.concat([ras, cas, mas, yas], ignore_index=True)

    l_node_count = len(l_nodes)
    i_node_count = len(i_nodes)

    new_loc_row = {'text':'Locutions', 'count':l_node_count, 'type':'Locution'}
    new_prop_row = {'text':'Propositions', 'count':i_node_count, 'type':'I-node'}

    overall_df = overall_df.append(new_loc_row, ignore_index=True)
    overall_df = overall_df.append(new_prop_row, ignore_index=True)

    dv = make_sunburst_chart(overall_df, ['type', 'text'], 'count', 'Overall Statistics' ,500, 500)

    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@app.route('/statistics-vis/<ids>', methods=["GET"])
def statistics_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)

    data = jsn['nodes']
    df_nodes = pd.DataFrame.from_dict(data, orient='columns')

    RA_nodes = df_nodes[df_nodes['type']=='RA']
    CA_nodes = df_nodes[df_nodes['type']=='CA']
    MA_nodes = df_nodes[df_nodes['type']=='MA']
    YA_nodes = df_nodes[df_nodes['type']=='YA']

    ras = RA_nodes['text'].value_counts().to_frame().reset_index()
    ras.columns = ['text', 'count']
    ras['type']='RA'

    cas = CA_nodes['text'].value_counts().to_frame().reset_index()
    cas.columns = ['text', 'count']
    cas['type']='CA'

    mas = MA_nodes['text'].value_counts().to_frame().reset_index()
    mas.columns = ['text', 'count']
    mas['type']='MA'

    yas = YA_nodes['text'].value_counts().to_frame().reset_index()
    yas.columns = ['text', 'count']
    yas['type']='YA'

    overall_df = pd.concat([ras, cas, mas, yas], ignore_index=True)

    l_node_count = len(l_nodes)
    i_node_count = len(i_nodes)

    new_loc_row = {'text':'Locutions', 'count':l_node_count, 'type':'Locution'}
    new_prop_row = {'text':'Propositions', 'count':i_node_count, 'type':'I-node'}

    overall_df = overall_df.append(new_loc_row, ignore_index=True)
    overall_df = overall_df.append(new_prop_row, ignore_index=True)

    dv = make_sunburst_chart(overall_df, ['type', 'text'], 'count', 'Overall Statistics' ,500, 500)

    response = app.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@app.route('/divisiveness-cloud-vis-view/<ids>', methods=["GET"])
def divisiveness_cloud_vis_view(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = centra.get_i_node_list(graph)
    divisiveness_list = get_divisiveness(graph, centra, i_nodes)

    df = pd.DataFrame(divisiveness_list, columns=['ID','Text', 'Divisiveness'])
    df = df.sort_values('Divisiveness', ascending=False)

    df_sel = df.head(10)


    plot_data = make_word_cloud(df_sel, "Text", 500, 500)


    return render_template('plot.html', plot_url=plot_data)

@app.route('/divisiveness-cloud-vis/<ids>', methods=["GET"])
def divisiveness_cloud_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = centra.get_i_node_list(graph)
    divisiveness_list = get_divisiveness(graph, centra, i_nodes)

    df = pd.DataFrame(divisiveness_list, columns=['ID','Text', 'Divisiveness'])
    df = df.sort_values('Divisiveness', ascending=False)

    df_sel = df.head(10)

    plot_data = make_word_cloud(df_sel, "Text", 500, 500)


    response = app.response_class(
        response=json.dumps(plot_data),
        status=200,
        mimetype='application/json'
    )
    return response

