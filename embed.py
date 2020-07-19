import time
import tensorflow as tf
from coarsen import create_coarse_graph
from utils import normalized, graph_to_adj
import numpy as np


def print_coarsen_info(ctrl, g):
    cnt = 0
    while g is not None:
        ctrl.logger.info("Level " + str(cnt) + " --- # nodes: " + str(g.node_num))
        g = g.coarser
        cnt += 1


def multilevel_embed(ctrl, graph, match_method, basic_embed, refine_model):
    '''This method defines the multilevel embedding method.'''

    start = time.time()

    # Step-1: Graph Coarsening.
    original_graph = graph
    coarsen_level = ctrl.coarsen_level
    if ctrl.refine_model.double_base:  # if it is double-base, it will need to do one more layer of coarsening
        coarsen_level += 1
    for i in range(coarsen_level):
        match, coarse_graph_size = match_method(ctrl, graph)
        coarse_graph = create_coarse_graph(ctrl, graph, match, coarse_graph_size)
        graph = coarse_graph
        if graph.node_num <= ctrl.embed_dim:
            ctrl.logger.error("Error: coarsened graph contains less than embed_dim nodes.")
            exit(0)

    if ctrl.debug_mode and graph.node_num < 1e3:
        assert np.allclose(graph_to_adj(graph).A, graph.A.A), "Coarser graph is not consistent with Adj matrix"
    print_coarsen_info(ctrl, original_graph)

    # Step-2 : Base Embedding
    if ctrl.refine_model.double_base:
        graph = graph.finer
    embedding = basic_embed(ctrl, graph)
    embedding = normalized(embedding, per_feature=False)

    # Step - 3: Embeddings Refinement.
    if ctrl.refine_model.double_base:
        coarse_embed = basic_embed(ctrl, graph.coarser)
        coarse_embed = normalized(coarse_embed, per_feature=False)
    else:
        coarse_embed = None
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=ctrl.workers)) as session:
        model = refine_model(ctrl, session)
        model.train_model(coarse_graph=graph.coarser, fine_graph=graph, coarse_embed=coarse_embed,
                          fine_embed=embedding)  # refinement model training

        while graph.finer is not None:  # apply the refinement model.
            embedding = model.refine_embedding(coarse_graph=graph, fine_graph=graph.finer, coarse_embed=embedding)
            graph = graph.finer

    end = time.time()
    ctrl.embed_time += end - start

    return embedding
