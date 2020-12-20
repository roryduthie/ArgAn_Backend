from flask import render_template, request, redirect, session, Markup, send_file
from . import application
import pandas as pd
import numpy as np
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


@application.route('/')
@application.route('/index')
def index():
    return redirect('/home')

@application.route('/home')
def test():
    data = {
        "user": "John Doe",
    }
    response = application.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response

def get_hevy_data(ids):
    ids = list(map(str,ids))
    datadict = {"mapIDs" : ids}
    headers = {
    'Content-Type': 'application/json',
    }
    data = json.dumps(datadict)
    response = requests.post('http://tomcat.arg.tech/ArgStructSearch/search/hevy/get/mapID', headers=headers, data=data)
    jsn = json.loads(response.text)
    return jsn

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


def get_peigen_cent(graph):

    centra = Centrality()

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    l_node_speakers = centra.get_l_node_speaker(graph, l_nodes)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    new_i_node_df = pd.DataFrame(new_i_nodes, columns=['ID', 'text', 'speaker'])
    new_i_node_df_sel = new_i_node_df[['ID', 'speaker']]
    i_nodes = centra.get_eigen_centrality(graph)
    i_node_df = pd.DataFrame(i_nodes, columns=['ID', 'Centrality', 'Text'])

    new_df = i_node_df.merge(new_i_node_df_sel, how='left', on='ID')


    return new_df

@application.route('/peigen-cent-raw/<ids>', methods=["GET"])
def peigen_cent_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes_df = get_peigen_cent(graph)

    data_dict = i_nodes_df .to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/eigen-cent-raw/<ids>', methods=["GET"])
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


    response = application.response_class(
        response=json.dumps(data_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/eigen-cent-vis-view/<ids>', methods=["GET"])
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

@application.route('/eigen-cent-vis/<ids>', methods=["GET"])
def eigen_cent_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)

    df = pd.DataFrame(i_nodes, columns=['id', 'cent', 'text'])
    df = df.sort_values('cent', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "text", "cent", "Eigenvector Centrality", 500, 500)


    response = application.response_class(
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

@application.route('/eigen-cent-cloud-vis-view/<ids>', methods=["GET"])
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

@application.route('/eigen-cent-cloud-vis/<ids>', methods=["GET"])
def eigen_cent_cloud_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)
    i_nodes = get_eigen_cent(graph)

    df = pd.DataFrame(i_nodes, columns=['id', 'cent', 'text'])
    df = df.sort_values('cent', ascending=False)
    df_sel = df.head(20)

    plot_data = make_word_cloud(df_sel, "text", 500, 500)


    response = application.response_class(
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

def get_pcogency(graph, centra):

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    l_node_speakers = centra.get_l_node_speaker(graph, l_nodes)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)


    yas = centra.get_ass_ya(graph)
    i_nodes_yas = centra.get_ya_i_nodes(graph, yas)

    ya_df = pd.DataFrame(i_nodes_yas, columns=['ID'])
    ya_df['count'] = 1



    i_node_ra_ca_list = centra.get_i_speaker_ra_ca_nodes(graph, new_i_nodes)

    i_node_df = pd.DataFrame(i_node_ra_ca_list, columns=['ID', 'speaker', 'RA', 'CA'])

    new_df = ya_df.merge(i_node_df, how='left', on='ID')
    new_df['speaker'] = new_df['speaker'].str.strip()

    sum_df = new_df.groupby(['speaker']).agg({'count':'sum','RA':'sum','CA':'sum'}).reset_index()

    sum_df['cogency'] = sum_df['RA'] / sum_df['count']

    sum_df_sel = sum_df[['speaker', 'cogency']]


    return sum_df_sel

@application.route('/pcogency-raw/<ids>', methods=["GET"])
def pcogency_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    cogency_df = get_pcogency(graph, centra)

    data_dict = cogency_df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/cogency-raw/<ids>', methods=["GET"])
def cogency_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    cogency = get_cogency(graph, centra)

    response = application.response_class(
        response=json.dumps(cogency),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/cogency-vis/<ids>', methods=["GET"])
def cogency_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    cogency = get_cogency(graph, centra)

    dv = make_gauge_chart(cogency, 'Cogency', None, 1)

    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@application.route('/cogency-vis-view/<ids>', methods=["GET"])
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

def get_pcorrectness(graph, centra):

    l_nodes = centra.get_l_node_list(graph)
    ta_nodes = centra.get_l_ta_nodes_count(graph, l_nodes)

    df_ta_nodes = pd.DataFrame(ta_nodes, columns=['text','Count'])
    df_l_nodes = pd.DataFrame(l_nodes, columns=['ID','text'])

    df_l_nodes['speaker'] = df_l_nodes['text'].str.split(':').str[0]
    df_ta_nodes['speaker'] = df_ta_nodes['text'].str.split(':').str[0]

    df_l_nodes['total'] = 1

    ta_nodes = df_ta_nodes.groupby(['speaker'])['Count'].agg('sum').to_frame().reset_index()
    ls = df_l_nodes.groupby(['speaker'])['total'].agg('sum').to_frame().reset_index()

    new_df = ls.merge(ta_nodes, how='left', on='speaker')
    new_df['Count'] = new_df['Count'].fillna(0)

    new_df['correctness'] = new_df['Count'] / new_df['total']

    new_df_sel = new_df[['speaker', 'correctness']]

    return new_df_sel

@application.route('/pcorrectness-raw/<ids>', methods=["GET"])
def pcorrectness_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    correctness_df = get_pcorrectness(graph, centra)

    data_dict = correctness_df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/correctness-raw/<ids>', methods=["GET"])
def correctness_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    correctness = get_correctness(graph, centra)

    response = application.response_class(
        response=json.dumps(correctness),
        status=200,
        mimetype='application/json'
    )
    return response


@application.route('/correctness-vis/<ids>', methods=["GET"])
def correctness_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    correctness = get_correctness(graph, centra)
    dv = make_gauge_chart(correctness, 'Correctness', None, 1)

    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response
@application.route('/correctness-vis-view/<ids>', methods=["GET"])
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

def get_p_cohrerence(graph, centra):

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    l_node_speakers = centra.get_l_node_speaker(graph, l_nodes)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)

    df_i_nodes = pd.DataFrame(new_i_nodes, columns=['ID','Text', 'Speaker'])
    df_i_nodes['Speaker'] = df_i_nodes['Speaker'].str.strip()

    graph = centra.remove_redundant_nodes(graph)
    isos = centra.get_isolated_nodes(graph)

    df_isolates = pd.DataFrame(isos, columns=['ID'])
    df_isolates['count'] = 1

    new_i_df = df_i_nodes.merge(df_isolates, how='left', on='ID')
    new_i_df['count'] = new_i_df['count'].fillna(0)

    new_i_df['Speaker'] = new_i_df['Speaker'].str.strip()
    isolates = new_i_df.groupby(['Speaker'])['count'].agg('sum').to_frame().reset_index()

    isolates['coherence'] = 1/isolates['count']
    isolates = isolates.replace([np.inf, -np.inf], np.nan)
    isolates['coherence'] =isolates['coherence'].fillna(0)

    isolates_sel = isolates[['Speaker', 'coherence']]




    return isolates_sel


@application.route('/pcoherence-raw/<ids>', methods=["GET"])
def pcoherence_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    coherence_df = get_p_cohrerence(graph, centra)

    data_dict = coherence_df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/coherence-raw/<ids>', methods=["GET"])
def coherence_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    coherence = get_coherence(graph, centra)

    response = application.response_class(
        response=json.dumps(coherence),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/coherence-vis/<ids>', methods=["GET"])
def coherence_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    coherence = get_coherence(graph, centra)

    dv = make_gauge_chart(coherence, 'Coherence', None, 1)

    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response
@application.route('/coherence-vis-view/<ids>', methods=["GET"])
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

def get_ppopularity(graph, centra):


    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    l_node_speakers = centra.get_l_node_speaker(graph, l_nodes)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)

    new_i_node_df = pd.DataFrame(new_i_nodes, columns=['ID', 'Text', 'speaker'])

    i_nodes = centra.get_i_node_ids(graph)
    yas, i_node_tups = centra.get_i_ya_nodes(graph, i_nodes)

    i_node_df = pd.DataFrame(i_node_tups, columns=['popularity', 'Text'])

    merge_df = i_node_df.merge(new_i_node_df, how='left', on='Text')
    pop_list = merge_df.to_dict(orient='records')
    return pop_list

def get_unpopularity(graph, centra):
    i_nodes = centra.get_i_node_ids(graph)
    yas, i_node_tups = centra.get_i_ya_dis_nodes(graph, i_nodes)
    return i_node_tups


@application.route('/unpopularity-raw/<ids>', methods=["GET"])
def unpopularity_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)

    response = application.response_class(
        response=json.dumps(popularity_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/unpopularity-vis-view/<ids>', methods=["GET"])
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

@application.route('/unpopularity-vis/<ids>', methods=["GET"])
def unpopularity_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)

    df = pd.DataFrame(popularity_list, columns=['UnPopularity','Text'])
    df = df.sort_values('UnPopularity', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "UnPopularity", "UnPopularity Top 10", 800, 500)


    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@application.route('/ppopularity-raw/<ids>', methods=["GET"])
def ppopularity_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_ppopularity(graph, centra)



    response = application.response_class(
        response=json.dumps(popularity_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/popularity-raw/<ids>', methods=["GET"])
def popularity_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)

    response = application.response_class(
        response=json.dumps(popularity_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/popularity-vis-view/<ids>', methods=["GET"])
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

@application.route('/popularity-vis/<ids>', methods=["GET"])
def popularity_vis(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)

    df = pd.DataFrame(popularity_list, columns=['Popularity','Text'])
    df = df.sort_values('Popularity', ascending=False)

    df_sel = df.head(10)

    dv = make_bar_chart(df_sel, "Text", "Popularity", "Popularity Top 10", 800, 500)


    response = application.response_class(
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

def get_pappeal(graph, centra, popularity_list):

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    popularity_2_list = centra.get_ra_ma_speaker_count(graph, new_i_nodes,centra)
    df1 = pd.DataFrame(popularity_list, columns =['Val', 'Text'])
    df2 = pd.DataFrame(popularity_2_list, columns =['Val', 'Text'])

    df1 = df1.set_index(['Text'])
    df2 = df2.set_index(['Text'])

    merge_df = df1.add(df2, fill_value=0)
    new_i_node_df = pd.DataFrame(new_i_nodes, columns=['ID', 'Text', 'speaker'])
    new_df = merge_df.merge(new_i_node_df, how='left', on='Text')

    appeal_list = new_df.to_dict(orient='records')
    return appeal_list

@application.route('/appeal-vis-view/<ids>', methods=["GET"])
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

@application.route('/appeal-vis/<ids>', methods=["GET"])
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


    response = application.response_class(
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



@application.route('/appeal-raw/<ids>', methods=["GET"])
def appeal_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)
    appeal_list = get_appeal(graph, centra, popularity_list)

    response = application.response_class(
        response=json.dumps(appeal_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/pappeal-raw/<ids>', methods=["GET"])
def pappeal_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_popularity(graph, centra)
    appeal_list = get_pappeal(graph, centra, popularity_list)

    response = application.response_class(
        response=json.dumps(appeal_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/unappeal-raw/<ids>', methods=["GET"])
def unappeal_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    popularity_list = get_unpopularity(graph, centra)
    appeal_list = get_unappeal(graph, centra, popularity_list)

    response = application.response_class(
        response=json.dumps(appeal_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/unappeal-vis-view/<ids>', methods=["GET"])
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

@application.route('/unappeal-vis/<ids>', methods=["GET"])
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


    response = application.response_class(
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

@application.route('/pdivisiveness-raw/<ids>', methods=["GET"])
def pdivisiveness_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    divisiveness_list = get_divisiveness(graph, centra, i_nodes)

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)

    div_df = pd.DataFrame(divisiveness_list, columns = ['ID', 'Text', 'Divisiveness'])
    i_node_df = pd.DataFrame(new_i_nodes, columns = ['ID', 'Text', 'Speaker'])

    i_node_df_sel = i_node_df[['ID', 'Speaker']]

    new_df = div_df.merge(i_node_df_sel, how='left', on='ID')

    div_list = new_df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(div_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/divisiveness-raw/<ids>', methods=["GET"])
def divisiveness_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    i_nodes = centra.get_i_node_list(graph)
    divisiveness_list = get_divisiveness(graph, centra, i_nodes)

    response = application.response_class(
        response=json.dumps(divisiveness_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/divisiveness-vis-view/<ids>', methods=["GET"])
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

@application.route('/divisiveness-vis/<ids>', methods=["GET"])
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


    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@application.route('/s-node-timeline-raw/<ids>', methods=["GET"])
def s_node_timeline_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_nodes = centra.get_l_node_list(graph)
    ta_timeline = centra.get_l_ta_s_nodes(graph, l_nodes)

    df = pd.DataFrame(ta_timeline, columns =['TA Count', 'Node Type', 'Node Count'])

    timeline_list = df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(timeline_list),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/s-node-timeline-vis/<ids>', methods=["GET"])
def s_node_timeline_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_nodes = centra.get_l_node_list(graph)
    ta_timeline = centra.get_l_ta_s_nodes(graph, l_nodes)

    df = pd.DataFrame(ta_timeline, columns =['TA Count', 'Node Type', 'Node Count'])

    dv = make_line_chart(df, 'TA Count', 'Node Count', 'Node Type', 'Node Type', 'Node Type', 600, 600)

    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@application.route('/s-node-timeline-vis-view/<ids>', methods=["GET"])
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

def make_treemap(dataframe, path):
    fig = px.treemap(dataframe,
        path=path
    )

    dv = plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
    return dv

@application.route('/hevy-hyp-evidence-vis-view/<ids>', methods=["GET"])
def hevy_hyp_evidence_vis_view(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    hypotheses_list = centra.get_hyp_evidence_nodes(graph, new_i_nodes)

    names = []
    parents = []

    for hyps in hypotheses_list:
        ID = hyps[0]
        text = hyps[1]
        speaker = hyps[2]
        evidence = hyps[3]

        if len(evidence) < 1:
            parents.append('')
            names.append(text)
        else:

            for evd in evidence:
                parents.append(text)
                names.append(evd)

    df = pd.DataFrame({'hypotheses':parents, 'evidence':names})

    dv = make_treemap(df, ['hypotheses', 'evidence'])

    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@application.route('/hevy-hyp-evidence-vis/<ids>', methods=["GET"])
def hevy_hyp_evidence_vis(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    hypotheses_list = centra.get_hyp_evidence_nodes(graph, new_i_nodes)

    names = []
    parents = []

    for hyps in hypotheses_list:
        ID = hyps[0]
        text = hyps[1]
        speaker = hyps[2]
        evidence = hyps[3]

        if len(evidence) < 1:
            parents.append(text)
            names.append('')
        else:

            for evd in evidence:
                parents.append(text)
                names.append(evd)

    df = pd.DataFrame({'hypotheses':parents, 'evidence':names})

    dv = make_treemap(df, ['hypotheses', 'evidence'])

    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

    return render_template('display_graph.html',
                               div_placeholder=Markup(dv)
                              )

@application.route('/hevy-hyp-evidence-raw/<ids>', methods=["GET"])
def hevy_hyp_evidence_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    hypotheses_list = centra.get_hyp_evidence_nodes(graph, new_i_nodes)
    hyp_df = pd.DataFrame(hypotheses_list, columns = ['ID', 'text', 'speaker', 'Evidence'])

    data_dict = hyp_df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/hevy-hyp-raw/<ids>', methods=["GET"])
def hevy_hyp_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)
    hypotheses_list = centra.get_hyp_i_nodes(graph, new_i_nodes)

    hyp_df = pd.DataFrame(hypotheses_list, columns = ['ID', 'text', 'speaker'])

    data_dict = hyp_df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/hevy-event-raw/<ids>', methods=["GET"])
def hevy_event_raw(ids):
    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)


    nodeset_list = []
    if not arg_map:
        data = load_nodesets_for_corpus(ids)
        nodeset_list = data['nodeSets']
    else:
        nodeset_list.append(ids)

    h_jsn = get_hevy_data(nodeset_list)
    event_list = get_event_info(h_jsn['nodes'])

    event_df = pd.DataFrame(event_list, columns = ['ID', 'Name', 'Agent', 'Object', 'Space', 'Time'])

    data_dict = event_df.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response


def get_event_info(nodes):

    event_list = []
    involved_agents = []
    involved_objects = []
    location_list = []
    timings_list = []
    for node in nodes:

        ntype = node['type']
        if ntype == 'Event':
            ID = node['nodeID']
            try:
                event_name = node['name']

            except:
                event_name = ''

            try:
                involved_agent = node['involvedAgent']

            except:
                involved_agent = ''

            try:
                involved_object = node['involved']

            except:
                involved_object = ''

            try:
                inSpace = node['inSpace']

            except:
                inSpace = ''

            try:
                circa = node['circa']

            except:
                circa = ''
            event_list.append((ID, event_name, involved_agent, involved_object, inSpace, circa))
    return event_list


@application.route('/statistics-raw/<ids>', methods=["GET"])
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

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/statistics-vis-view/<ids>', methods=["GET"])
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

@application.route('/statistics-vis/<ids>', methods=["GET"])
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

    response = application.response_class(
        response=dv,
        status=200,
        mimetype='application/html'
    )
    return response

@application.route('/divisiveness-cloud-vis-view/<ids>', methods=["GET"])
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

@application.route('/divisiveness-cloud-vis/<ids>', methods=["GET"])
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


    response = application.response_class(
        response=json.dumps(plot_data),
        status=200,
        mimetype='application/json'
    )
    return response

def get_sycophancy(graph, centra, l_nodes, l_node_speakers):
    speaker_agreements = centra.get_agreement_for_speaker(graph, l_nodes)
    df_agreements = pd.DataFrame(speaker_agreements, columns=['speaker', 'agreement'])
    df_locs = pd.DataFrame(l_node_speakers, columns=['text', 'speaker'])
    locs = df_locs['speaker'].str.strip().value_counts().to_frame().reset_index()
    locs.columns = ['speaker', 'count']
    agrees = df_agreements.groupby(['speaker'])['agreement'].agg('sum').to_frame().reset_index()
    agrees['speaker'] = agrees['speaker'].str.strip()

    new_df = agrees.merge(locs, how='left', on = ['speaker'])

    new_df['sycophancy'] = new_df['agreement'] / new_df['count']
    return new_df, agrees





@application.route('/sycophancy-raw/<ids>', methods=["GET"])
def sycophancy_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_nodes = centra.get_l_node_list(graph)
    l_node_speakers = centra.get_l_node_speaker(graph, l_nodes)

    syc_df, agrees_df = get_sycophancy(graph, centra, l_nodes, l_node_speakers)
    if syc_df.empty:
        response = application.response_class(
            response=json.dumps('No Agreements'),
            status=200,
            mimetype='application/json'
        )
        return response
    syc_df_sel = syc_df[['speaker', 'sycophancy']]


    data_dict = syc_df_sel.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

def get_idolatry(agreement_pairs, interaction_pairs):



    df_agreement_pairs = pd.DataFrame(agreement_pairs, columns=['speaker1','speaker2', 'agreement'])
    df_interactions = pd.DataFrame(interaction_pairs, columns=['speaker1', 'speaker2', 'interaction'])

    df_agreement_pairs[['speaker1', 'speaker2']] = np.sort(df_agreement_pairs[['speaker1', 'speaker2']], axis=1)
    df_interactions[['speaker1', 'speaker2']] = np.sort(df_interactions[['speaker1', 'speaker2']], axis=1)

    agrees = df_agreement_pairs.groupby(['speaker1','speaker2'])['agreement'].agg('sum').to_frame().reset_index()
    inters = df_interactions.groupby(['speaker1','speaker2'])['interaction'].agg('sum').to_frame().reset_index()

    new_df = agrees.merge(inters, how='left', on = ['speaker1', 'speaker2'])
    new_df['interaction'] = new_df['interaction'].fillna(1)

    new_df['idolatry'] = new_df['agreement'] / new_df['interaction']
    return new_df





@application.route('/idolatry-raw/<ids>', methods=["GET"])
def idolatry_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)

    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)

    agreement_pairs = centra.get_agreement_speaker_pair_count(graph, new_i_nodes)
    interaction_pairs = centra.get_interactions(graph, l_nodes)



    idol_df = get_idolatry(agreement_pairs, interaction_pairs)
    if idol_df.empty:
        response = application.response_class(
            response=json.dumps('No Agreements'),
            status=200,
            mimetype='application/json'
        )
        return response
    idol_df_sel = idol_df[['speaker1','speaker2', 'idolatry']]


    data_dict = idol_df_sel.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response





@application.route('/interactions-raw/<ids>', methods=["GET"])
def interaction_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)


    l_nodes = centra.get_l_node_list(graph)


    interaction_pairs = centra.get_interactions(graph, l_nodes)
    df_interactions = pd.DataFrame(interaction_pairs, columns=['speaker1', 'speaker2', 'interaction'])
    df_interactions[['speaker1', 'speaker2']] = np.sort(df_interactions[['speaker1', 'speaker2']], axis=1)
    inters = df_interactions.groupby(['speaker1','speaker2'])['interaction'].agg('sum').to_frame().reset_index()



    if inters.empty:
        response = application.response_class(
            response=json.dumps('No Interactions'),
            status=200,
            mimetype='application/json'
        )
        return response
    inters_df_sel = inters[['speaker1','speaker2', 'interaction']]


    data_dict = inters_df_sel.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/responsiveness-raw/<ids>', methods=["GET"])
def responsiveness_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)


    yas = centra.get_ya_nodes_list(graph)


    responses_list = centra.get_responsiveness(graph, yas)
    df_resps = pd.DataFrame(responses_list, columns=['speaker', 'questions', 'answers'])
    resps = df_resps.groupby(['speaker']).agg({'questions':'sum','answers':'sum'}).reset_index()
    resps['responsiveness'] = resps['answers'] / resps['questions']


    if resps.empty:
        response = application.response_class(
            response=json.dumps('No Questions'),
            status=200,
            mimetype='application/json'
        )
        return response
    resps_df_sel = resps[['speaker','responsiveness']]


    data_dict = resps_df_sel.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response



def get_belligerence(graph, centra, l_nodes, l_node_speakers, i_node_speakers):

    speaker_cas = centra.get_speaker_ca_nodes(graph, i_node_speakers)
    df_cas = pd.DataFrame(speaker_cas, columns=['speaker', 'conflict'])
    df_locs = pd.DataFrame(l_node_speakers, columns=['text', 'speaker'])
    locs = df_locs['speaker'].str.strip().value_counts().to_frame().reset_index()
    locs.columns = ['speaker', 'count']
    cas = df_cas.groupby(['speaker'])['conflict'].agg('sum').to_frame().reset_index()
    cas['speaker'] = cas['speaker'].str.strip()

    new_df = cas.merge(locs, how='left', on = ['speaker'])

    new_df['belligerence'] = new_df['conflict'] / new_df['count']
    return new_df





@application.route('/belligerence-raw/<ids>', methods=["GET"])
def belligerence_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)



    l_node_i_node_list = centra.get_loc_prop_pair(graph)
    i_nodes = centra.get_i_node_list(graph)
    l_nodes = centra.get_l_node_list(graph)

    l_node_speakers = centra.get_l_node_speaker(graph, l_nodes)
    new_i_nodes = centra.get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra)

    bell_df = get_belligerence(graph, centra, l_nodes, l_node_speakers, new_i_nodes)
    if bell_df.empty:
        response = application.response_class(
            response=json.dumps('No Conflict'),
            status=200,
            mimetype='application/json'
        )
        return response
    bell_df_sel = bell_df[['speaker', 'belligerence']]


    data_dict = bell_df_sel.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@application.route('/stimulating-raw/<ids>', methods=["GET"])
def stimulating_raw(ids):

    arg_map = is_map(ids)
    centra = Centrality()
    graph, jsn = get_graph_jsn(ids, arg_map)

    l_nodes = centra.get_l_node_list(graph)
    l_node_speakers = centra.get_l_node_speaker(graph, l_nodes)

    df_locs = pd.DataFrame(l_node_speakers, columns=['text', 'speaker'])
    locs = df_locs['speaker'].str.strip().value_counts().to_frame().reset_index()
    locs.columns = ['speaker', 'count']

    syc_df, agrees_df = get_sycophancy(graph, centra, l_nodes, l_node_speakers)


    interaction_pairs = centra.get_interactions(graph, l_nodes)
    df_interactions = pd.DataFrame(interaction_pairs, columns=['speaker1', 'speaker2', 'interaction'])
    sel_inters = df_interactions[['speaker1', 'interaction']]
    sel_inters['speaker1'] = sel_inters['speaker1'].str.strip()

    single_inters = sel_inters.groupby(['speaker1'])['interaction'].agg('sum').to_frame().reset_index()
    single_inters.columns = ['speaker', 'interaction']

    overall_df = single_inters.merge(agrees_df, how='left', on=['speaker'])
    overall_df['agreement'] = overall_df['agreement'].fillna(1)

    overall_df['total'] = overall_df['interaction'] + overall_df['agreement']

    new_df = overall_df.merge(locs, how='left', on = ['speaker'])

    new_df['stimulating'] = new_df['total'] / new_df['count']

    if new_df.empty:
        response = application.response_class(
            response=json.dumps('No stimulating'),
            status=200,
            mimetype='application/json'
        )
        return response
    new_df_sel = new_df[['speaker', 'stimulating']]


    data_dict = new_df_sel.to_dict(orient='records')

    response = application.response_class(
        response=json.dumps(data_dict),
        status=200,
        mimetype='application/json'
    )
    return response


