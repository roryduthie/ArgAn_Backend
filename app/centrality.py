from .load_map import CorpusLoader
from . import app
import json
import requests
from datetime import datetime
from pathlib import Path
import re
import networkx as nx
import urllib.request



class Centrality:
    @staticmethod
    def get_nodeset_path(nodeset_id):
        corpus_name = 'US2016tv'
        directory_path = 'examples/'
        node_path = directory_path + 'nodeset' + nodeset_id + '.json'
        return node_path

    @staticmethod
    def get_svg_path(nodeset_id):
        #corpus_name = 'US2016tv'
        directory_path = 'examples/'
        node_path = directory_path + nodeset_id + '.svg'
        return node_path

    @staticmethod
    def create_svg_url(nodeset_id, isMap):

        if isMap:
            return 'http://www.aifdb.org/diagram/svg/' + nodeset_id
        else:
            return 'http://corpora.aifdb.org/' + nodeset_id + '/svg/'
        return node_path

    @staticmethod
    def create_json_url(nodeset_id, isMap):

        if isMap:
            return 'http://www.aifdb.org/json/' + nodeset_id
        else:
            return 'http://corpora.aifdb.org/' + nodeset_id + '/json/'
        return node_path
    @staticmethod
    def get_graph_string(json_file):
        corpus_loader = CorpusLoader()
        try:
            graph = corpus_loader.parse_json(json_file)
        except(IOError):
            print('File was not found:')
            print(node_path)

        return graph

    @staticmethod
    def get_graph(node_path):
        corpus_loader = CorpusLoader()
        try:
            with open(node_path) as json_data:
                graph = corpus_loader.parse_json(json.load(json_data))
        except(IOError):
            print('File was not found:')
            print(node_path)

        return graph

    @staticmethod
    def get_graph_url(node_path):
        corpus_loader = CorpusLoader()
        try:
            jsn_string = requests.get(node_path).text
            strng_ind = jsn_string.index('{')
            n_string = jsn_string[strng_ind:]
            dta = json.loads(n_string)
            graph = corpus_loader.parse_json(dta)
        except(IOError):
            print('File was not found:')
            print(node_path)

        return graph, dta

    @staticmethod
    def remove_redundant_nodes(graph):

        node_types=nx.get_node_attributes(graph,'type')

        nodes_to_remove =  [x for x,y in graph.nodes(data=True) if y['type']=='TA' or y['type']=='L' or y['type']=='YA']

        graph.remove_nodes_from(nodes_to_remove)

        return graph

    @staticmethod
    def remove_an_nodes(graph):

        node_types=nx.get_node_attributes(graph,'type')

        nodes_to_remove =  [x for x,y in graph.nodes(data=True) if y['type']=='YA' and y['text']=='Analysing']

        graph.remove_nodes_from(nodes_to_remove)

        return graph

    @staticmethod
    def remove_redundant_nodes_not_ya(graph):

        node_types=nx.get_node_attributes(graph,'type')

        nodes_to_remove =  [x for x,y in graph.nodes(data=True) if y['type']=='TA' or y['type']=='L']

        graph.remove_nodes_from(nodes_to_remove)

        return graph
    @staticmethod
    def remove_iso_nodes(graph):
        graph.remove_nodes_from(list(nx.isolates(graph)))
        return graph
    @staticmethod
    def remove_iso_analyst_nodes(graph):
        analyst_nodes = []
        isolated_nodes = list(nx.isolates(graph))
        for node in isolated_nodes:
            if graph.nodes[node]['type'] == 'L':
                analyst_nodes.append(node)
        graph.remove_nodes_from(analyst_nodes)
        return graph

    @staticmethod
    def get_eigen_centrality(graph):
        try:
            cent = nx.eigenvector_centrality_numpy(graph)
        except:
            cent = nx.degree_centrality(graph)

        nx.set_node_attributes(graph, cent, 'central')
        i_nodes =  [(x,y['central'],y['text']) for x,y in graph.nodes(data=True) if y['type']=='I']
        return i_nodes

    @staticmethod
    def sort_by_centrality(i_nodes):
        sorted_by_second = sorted(i_nodes, key=lambda tup: tup[1])
        ordered_ids = [(i[0],i[2]) for i in sorted_by_second]

        return ordered_ids

    @staticmethod
    def list_nodes(graph):
        return list(graph)

    @staticmethod
    def get_s_node_list(graph):
        s_nodes =  [x for x,y in graph.nodes(data=True) if y['type']=='MA' or y['type']=='RA' or y['type']=='CA' or y['type']=='PA']
        return s_nodes


    @staticmethod
    def get_l_node_list(graph):
        l_nodes =  [(x,y['text']) for x,y in graph.nodes(data=True) if y['type']=='L']
        return l_nodes

    @staticmethod
    def get_extended_l_node_list(graph, nodeset_id):
        l_nodes =  [(x,y['text'], str(y['timestamp']), nodeset_id) for x,y in graph.nodes(data=True) if y['type']=='L']
        return l_nodes
    @staticmethod
    def get_i_node_list(graph):
        i_nodes =  [(x,y['text']) for x,y in graph.nodes(data=True) if y['type']=='I']
        return i_nodes
    @staticmethod
    def get_i_node_ids(graph):
        i_nodes =  [(x) for x,y in graph.nodes(data=True) if y['type']=='I']
        return i_nodes
    @staticmethod
    def get_extended_i_node_list(graph, nodeset_id):
        i_nodes =  [(x,y['text'], nodeset_id) for x,y in graph.nodes(data=True) if y['type']=='I']
        return i_nodes

    @staticmethod
    def get_divergent_nodes(graph):
        list_of_nodes = []

        for v in list(graph.nodes):
            node_pres = []
            node_pres = list(graph.successors(v))
            if len(node_pres) > 1:
                list_of_nodes.append(v)
        return list_of_nodes

    @staticmethod
    def get_loc_prop_pair(graph):
        i_node_ids =  [x for x,y in graph.nodes(data=True) if y['type']=='I']
        locution_prop_pair = []
        for node_id in i_node_ids:
            preds = list(graph.predecessors(node_id))
            for pred in preds:
                node_type=graph.nodes[pred]['type']
                node_text = graph.nodes[pred]['text']

                if node_type == 'YA' and node_text != 'Agreeing':
                    ya_preds = list(graph.predecessors(pred))
                    for ya_pred in ya_preds:
                        pred_node_type=graph.nodes[ya_pred]['type']
                        pred_node_text=graph.nodes[ya_pred]['text']

                        if pred_node_type == 'L':
                            locution_prop_pair.append((ya_pred, node_id))
        return locution_prop_pair

    @staticmethod
    def get_child_edges(graph):
        list_of_nodes = []
        list_of_edges = []

        for v in list(graph.nodes):
            node_pres = []
            node_pres = list(nx.ancestors(graph, v))
            list_of_nodes.append((v, node_pres))
            edges = []
            edges = list(nx.edge_dfs(graph,v, orientation='reverse'))
            res_list = []
            res_list = [(x[0], x[1]) for x in edges]
            list_of_edges.append((v, res_list))

        return list_of_nodes, list_of_edges
    @staticmethod
    def get_top_nodes_combined(node_list):
        all_nodes = []
        centra = Centrality()
        G = nx.DiGraph()
        for node in node_list:
            dir_path = 'http://www.aifdb.org/json/' + str(node)
            g1 = centra.get_graph_url(dir_path)

            G = nx.compose(G,g1)
        G = centra.remove_iso_nodes(G)
        l_nodes = centra.get_l_node_list(G)
        l_node_i_node = centra.get_loc_prop_pair(G)
        g = centra.remove_redundant_nodes(G)
        i_nodes = centra.get_eigen_centrality(g)
        sorted_nodes = centra.sort_by_centrality(i_nodes)
        #print(len(i_nodes))
        #ten_percent = 0.1 * len(i_nodes)
        #top_n = sorted_nodes[:int(ten_percent)]
        all_nodes.extend(sorted_nodes)

        if len(all_nodes) > 10:
            ten_percent = 0.05 * len(all_nodes)
        else:
            return all_nodes, l_node_i_node, l_nodes

        return all_nodes[:int(round(ten_percent))], l_node_i_node, l_nodes

    @staticmethod
    def get_all_nodes_combined(node_list):
        all_nodes = []
        G = nx.DiGraph()
        centra = Centrality()
        for node in node_list:
            dir_path = 'http://www.aifdb.org/json/' + str(node)
            g1 = centra.get_graph_url(dir_path)

            G = nx.compose(G,g1)
        G = centra.remove_iso_nodes(G)
        l_nodes = centra.get_l_node_list(G)
        l_node_i_node = centra.get_loc_prop_pair(G)
        g = centra.remove_redundant_nodes(G)
        i_nodes = centra.get_eigen_centrality(g)
        sorted_nodes = centra.sort_by_centrality(i_nodes)
        #print(len(i_nodes))
        #ten_percent = 0.1 * len(i_nodes)
        #top_n = sorted_nodes[:int(ten_percent)]

        return sorted_nodes, l_node_i_node, l_nodes

    @staticmethod
    def get_ras(graph):
        ra_nodes =  [x for x,y in graph.nodes(data=True) if y['type']=='RA']
        return ra_nodes

    @staticmethod
    def get_ra_i_nodes(graph, ras):
        ra_tups = []
        for ra in ras:
            node_succ = list(graph.successors(ra))
            i_1 = node_succ[0]
            i_1_text = graph.nodes[i_1]['text']
            node_pres = list(graph.predecessors(ra))

            for n in node_pres:
                n_type = graph.nodes[n]['type']
                if n_type == 'I':
                    i_2 = n
                    i_2_text = graph.nodes[i_2]['text']
                    break

            ra_tup = (ra, i_1_text, i_2_text)
            ra_tups.append(ra_tup)
        return ra_tups

    @staticmethod
    def get_ass_ya(graph):
        ya_nodes =  [x for x,y in graph.nodes(data=True) if y['type']=='YA' and y['text']=='Asserting' or y['type']=='YA' and y['text']=='Hypothesising']
        return ya_nodes

    @staticmethod
    def get_TAs(graph):
        ta_nodes =  [x for x,y in graph.nodes(data=True) if y['type']=='TA']
        return ta_nodes

    @staticmethod
    def get_ya_i_nodes(graph, yas):
        ya_i = []
        for ya in yas:
            node_succ = list(graph.successors(ya))
            i_1 = node_succ[0]
            ya_i.append(i_1)
        return ya_i

    @staticmethod
    def get_i_ra_ca_nodes(graph, i_nodes):
        ra_list = []
        ca_list = []
        for i in i_nodes:
            node_succ = list(graph.predecessors(i))

            for n in node_succ:
                n_type = graph.nodes[n]['type']
                if n_type == 'RA':
                    ra_list.append(n)
                if n_type == 'CA':
                    ca_list.append(n)
        return ra_list, ca_list
    @staticmethod
    def get_i_ya_nodes(graph, i_nodes):
        ya_list = []
        i_count = []

        for i in i_nodes:
            node_succ = list(graph.predecessors(i))
            ya_count = 0
            i_text = graph.nodes[i]['text']
            for n in node_succ:
                n_type = graph.nodes[n]['type']
                n_text = graph.nodes[n]['text']
                if n_type == 'YA' and n_text == 'Asserting' or n_type == 'YA' and n_text == 'Agreeing' or n_type == 'YA' and n_text == 'Hypothesising':
                    ya_list.append(n)
                    ya_count = ya_count + 1

            i_count_tup = (ya_count, i_text)
            i_count.append(i_count_tup)
        return ya_list, i_count

    @staticmethod
    def get_i_ya_dis_nodes(graph, i_nodes):
        ya_list = []
        i_count = []

        for i in i_nodes:
            node_succ = list(graph.predecessors(i))
            ya_count = 0
            i_text = graph.nodes[i]['text']
            for n in node_succ:
                n_type = graph.nodes[n]['type']
                n_text = graph.nodes[n]['text']
                if n_type == 'YA' and n_text == 'Disagreeing':
                    ya_list.append(n)
                    ya_count = ya_count + 1

            i_count_tup = (ya_count, i_text)
            i_count.append(i_count_tup)
        return ya_list, i_count

    @staticmethod
    def get_l_ta_nodes(graph, l_nodes):
        ta_list = []
        for lnode in l_nodes:
            l = lnode[0]
            node_succ = list(graph.predecessors(l))
            for n in node_succ:
                n_type = graph.nodes[n]['type']
                if n_type == 'TA':
                    ta_list.append(n)
                    break
        return ta_list
    @staticmethod
    def get_isolated_nodes(graph):
        isos = list(nx.isolates(graph))
        return isos
    @staticmethod
    def get_l_node_text(i_node_id, lnode_inode_list, l_node_list):
        for rel_tup in lnode_inode_list:
            lnode_id = rel_tup[0]
            inode_id = rel_tup[1]

            if i_node_id == inode_id:
                for tups in l_node_list:
                    l_id = tups[0]
                    if l_id == lnode_id:
                        ltext = tups[1]
                        return l_id, ltext
    @staticmethod
    def get_i_node_speaker_list(i_nodes, l_nodes, l_node_i_node_list,centra):
        new_i_nodes = []
        for i_node in i_nodes:
            ID = i_node[0]
            text = i_node[1]
            l_id, l_text = centra.get_l_node_text(ID, l_node_i_node_list, l_nodes)
            speaker = l_text.split(':')[0]
            new_i_node = (ID, text, speaker)
            new_i_nodes.append(new_i_node)
        return new_i_nodes

    @staticmethod
    def get_i_node_speaker(ID,i_node_speaker_list):
        for i_node in i_node_speaker_list:
            nodeID = i_node[0]
            text = i_node[1]
            speaker = i_node[2]

            if ID == nodeID:
                return speaker
    @staticmethod
    def get_ra_ma_speaker_count(graph, i_nodes_speak, centra):
        i_count = []
        for i_node in i_nodes_speak:
            ID = i_node[0]
            text = i_node[1]
            speaker = i_node[2]
            count = 0
            node_pres = list(graph.predecessors(ID))
            for n in node_pres:
                n_type = graph.nodes[n]['type']
                if n_type == 'RA' or n_type == 'MA':
                    i_node_pres = list(graph.predecessors(n))
                    for node in i_node_pres:
                        node_type = graph.nodes[node]['type']
                        if node_type == 'I':
                            speaker_2 = centra.get_i_node_speaker(node,i_nodes_speak)
                            if speaker != speaker_2:
                                count = count + 1
            i_count_tup = (count, text)
            i_count.append(i_count_tup)
        return i_count

    @staticmethod
    def get_ca_ma_speaker_count(graph, i_nodes_speak, centra):
        i_count = []
        for i_node in i_nodes_speak:
            ID = i_node[0]
            text = i_node[1]
            speaker = i_node[2]
            count = 0
            node_pres = list(graph.predecessors(ID))
            for n in node_pres:
                n_type = graph.nodes[n]['type']
                if n_type == 'CA' or n_type == 'MA':
                    i_node_pres = list(graph.predecessors(n))
                    for node in i_node_pres:
                        node_type = graph.nodes[node]['type']
                        if node_type == 'I':
                            speaker_2 = centra.get_i_node_speaker(node,i_nodes_speak)
                            if speaker != speaker_2:
                                count = count + 1
            i_count_tup = (count, text)
            i_count.append(i_count_tup)
        return i_count

    @staticmethod
    def get_ca_i_nodes(graph, ca_list):
        ca_ra_list = []
        for ca in ca_list:
            node_succ = list(graph.predecessors(ca))
            ra_list = []
            for n in node_succ:
                node_succ_2 = list(graph.predecessors(n))
                for node in node_succ_2:
                    n_type = graph.nodes[node]['type']
                    if n_type == 'RA':
                        ra_list.append(n)
            ca_ra_tup = (ca, ra_list)
            ca_ra_list.append(ca_ra_tup)
        return ca_ra_list

    @staticmethod
    def get_i_ca_nodes(graph, centra, i_node_id):

        ca_list = []
        ra_list = []
        ca_ra_list = []
        node_pred = list(graph.predecessors(i_node_id))
        node_succ = list(graph.successors(i_node_id))

        for n in node_pred:
            n_type = graph.nodes[n]['type']
            if n_type == 'RA':
                ra_list.append(n)
            if n_type == 'CA':
                ca_list.append(n)
        for n in node_succ:
            n_type = graph.nodes[n]['type']
            if n_type == 'CA':
                ca_list.append(n)
        if len(ca_list) > 0:
            ca_ra_list = centra.get_ca_i_nodes(graph, ca_list)

        return ra_list, ca_list, ca_ra_list

    @staticmethod
    def get_l_ta_s_nodes(graph, l_nodes):
        ta_list = []
        ra_count = 0
        ca_count = 0
        ma_count = 0
        ta_count = 0

        for lnode in l_nodes:
            l = lnode[0]
            node_succ = list(graph.successors(l))
            for n in node_succ:
                n_type = graph.nodes[n]['type']
                if n_type == 'TA':
                    ta_count = ta_count +1
                    node_pred = list(graph.successors(n))
                    for node in node_pred:
                        node_type = graph.nodes[node]['type']
                        node_text = graph.nodes[node]['text']

                        if node_type == 'YA' and node_text == 'Arguing':
                            ra_count = ra_count +1
                        if node_type == 'YA' and node_text == 'Restating':
                            ma_count = ma_count +1
                        if node_type == 'YA' and node_text == 'Disagreeing':
                            ca_count = ca_count +1
                ra_tup = (ta_count, "RA", ra_count)
                ma_tup = (ta_count, "MA", ma_count)
                ca_tup = (ta_count, "CA", ca_count)
                ta_list.append(ra_tup)
                ta_list.append(ma_tup)
                ta_list.append(ca_tup)

        return ta_list
