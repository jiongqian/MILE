from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import logging
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sys
from graph import Graph

class Mapping:
    '''Used for mapping index of nodes since the data structure used for graph requires continuous index.'''
    def __init__(self, old2new, new2old):
        self.old2new = old2new
        self.new2old = new2old

def read_graph(ctrl, file_path, edgelist=False, metis=False):
    '''Returns an instance of Graph (undirected) and a index mapping dict if any.'''
    assert edgelist or metis, "Needs to specify the format of the input graph."
    if edgelist:
        return _read_graph_from_edgelist(ctrl, file_path)
    return _read_graph_from_metis(ctrl, file_path)

def _read_graph_from_edgelist(ctrl, file_path):
    '''Assume each edge shows up ONLY once: small-id<space>large-id, or small-id<space>large-id<space>weight. 
    Indices are not required to be continuous.'''
#    logging.info("Reading graph from edgelist...")
    in_file = open(file_path)
    neigh_dict = defaultdict(list)
    max_idx = -1
    edge_num = 0
    for line in in_file:
        eles = line.strip().split()
	n0, n1 = [int(ele) for ele in eles[:2]]
	assert n0 <= n1, "first id in a row should be the smaller one..."
        if len(eles) == 3: # weighted graph
            wgt = float(eles[2])
            neigh_dict[n0].append((n1, wgt))
            if n0 != n1:
                neigh_dict[n1].append((n0, wgt))
        else:
            neigh_dict[n0].append(n1)
            if n0 != n1:
                neigh_dict[n1].append(n0)
        if n0 != n1:
            edge_num += 2
        else:
            edge_num += 1
        max_idx = max(max_idx, n1)
    in_file.close()
    weighted = (len(eles) == 3)
    continuous_idx = (max_idx+1 == len(neigh_dict)) # starting from zero
    mapping = None
    if not continuous_idx:
        old2new = dict()
        new2old = dict()
        cnt = 0
        sorted_keys = sorted(neigh_dict.keys())
        for key in sorted_keys:
            old2new[key] = cnt
            new2old[cnt] = key
            cnt += 1
        new_neigh_dict = defaultdict(list)
        for key in sorted_keys:
            for tpl in neigh_dict[key]:
                node_u = old2new[key]
                if weighted:
                    new_neigh_dict[node_u].append((old2new[tpl[0]], tpl[1]))
                else:
                    new_neigh_dict[node_u].append(old2new[tpl])
        del sorted_keys
        neigh_dict = new_neigh_dict # remapped
        mapping = Mapping(old2new, new2old)

    node_num = len(neigh_dict)
    graph = Graph(node_num, edge_num)
    edge_cnt = 0
    graph.adj_idx[0] = 0
    for idx in range(node_num):
        graph.node_wgt[idx] = 1 # default weight to nodes
        for neigh in neigh_dict[idx]:
            if weighted:
                graph.adj_list[edge_cnt] = neigh[0]
                graph.adj_wgt[edge_cnt] = neigh[1]
            else:
                graph.adj_list[edge_cnt] = neigh
                graph.adj_wgt[edge_cnt] = 1.0
            edge_cnt += 1
        graph.adj_idx[idx+1] = edge_cnt

    if ctrl.debug_mode:
        assert nx.is_connected(graph2nx(graph)), "Only single connected component is allowed for embedding."
    
    graph.A = graph_to_adj(graph, self_loop=False)
    return graph, mapping

def _read_graph_from_metis(ctrl, file_path):
    '''Assume idx starts from *1* and are continuous. Edge shows up twice. Assume single connected component.'''
#    logging.info("Reading graph from metis...")
    in_file = open(file_path)
    first_line = [int(ele) for ele in in_file.readline().strip().split()]
    weighted = False 
    if len(first_line) == 3 and first_line[-1] == 1:
        weighted = True
    node_num, edge_num = first_line[:2]
    edge_num *= 2
    graph = Graph(node_num, edge_num)
    edge_cnt = 0
    graph.adj_idx[0] = 0
    for idx in range(node_num):
        graph.node_wgt[idx] = 1
        eles = in_file.readline().strip().split()
        j = 0 
        while j < len(eles):
            neigh = int(eles[j]) - 1 # needs to minus 1 as metis starts with 1.
            if weighted:
                wgt = float(eles[j+1])
            else:
                wgt = 1.0
            graph.adj_list[edge_cnt] = neigh # self-loop included.
            graph.adj_wgt[edge_cnt] = wgt
            graph.degree[idx] += wgt
            edge_cnt += 1
            if weighted:
                j += 2
            else:
                j += 1
        graph.adj_idx[idx+1] = edge_cnt
    graph.A = graph_to_adj(graph, self_loop=False)
    # check connectivity in debug mode
    if ctrl.debug_mode:
        assert nx.is_connected(graph2nx(graph))

    return graph, None

def graph2nx(graph): # mostly for debugging purpose. weights ignored.
    G=nx.Graph()
    for idx in range(graph.node_num):
        for neigh_idx in range(graph.adj_idx[idx], graph.adj_idx[idx+1]):
            neigh = graph.adj_list[neigh_idx]
            if neigh>idx:
                G.add_edge(idx, neigh)
    return G

def graph_to_adj(graph, self_loop=False):
    '''self_loop: manually add self loop or not'''
    node_num = graph.node_num
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(0, node_num):
        for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i+1]):
            i_arr.append(i)
            j_arr.append(graph.adj_list[neigh_idx])
            data_arr.append(graph.adj_wgt[neigh_idx])
    adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


def cmap2C(cmap): # fine_graph to coarse_graph, matrix format of cmap: C: n x m, n>m.
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)
    return sp.csr_matrix((data_arr, (i_arr, j_arr)))        

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)
    return logger

def normalized(embeddings, per_feature=True):
    if per_feature:
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaler.fit(embeddings)
        return scaler.transform(embeddings)
    else:
        return normalize(embeddings, norm='l2')
