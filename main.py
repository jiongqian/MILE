#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from coarsen import generate_hybrid_matching
from defs import Control
from embed import multilevel_embed
from eval_embed import eval_multilabel_clf
from refine_model import GCN, GraphSage
from utils import read_graph, setup_custom_logger
import importlib
import logging
import numpy as np
import tensorflow as tf


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=False, default='PPI',
                        help='Input graph file')
    parser.add_argument('--format', required=False, default='metis', choices=['metis', 'edgelist'],
                        help='Format of the input graph file (metis/edgelist)')
    parser.add_argument('--store-embed', action='store_true',
                        help='Store the embeddings.')
    parser.add_argument('--no-eval', action='store_true',
                        help='Evaluate the embeddings.')
    parser.add_argument('--embed-dim', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--basic-embed', required=False, default='netmf',
                        choices=['deepwalk', 'grarep', 'netmf'],
                        help='The basic embedding method. If you added a new embedding method, please add its name to choices')
    parser.add_argument('--refine-type', required=False, default='MD-gcn',
                        choices=['MD-gcn', 'MD-dumb', 'MD-gs'],
                        help='The method for refining embeddings.')
    parser.add_argument('--coarsen-level', default=2, type=int,
                        help='MAX number of levels of coarsening.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers.')
    parser.add_argument('--double-base', action='store_true',
                        help='Use double base for training')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Learning rate of the refinement model')
    parser.add_argument('--self-weight', default=0.05, type=float,
                        help='Self-loop weight for GCN model.')  # usually in the range [0, 1]
    # Consider increasing self-weight a little bit if coarsen-level is high.
    args = parser.parse_args()
    return args


def set_control_params(ctrl, args, graph):
    ctrl.refine_model.double_base = args.double_base
    ctrl.refine_model.learning_rate = args.learning_rate
    ctrl.refine_model.self_weight = args.self_weight

    ctrl.coarsen_level = args.coarsen_level
    ctrl.coarsen_to = max(1, graph.node_num // (2 ** ctrl.coarsen_level))  # rough estimation.
    ctrl.embed_dim = args.embed_dim
    ctrl.basic_embed = args.basic_embed
    ctrl.refine_type = args.refine_type
    ctrl.data = args.data
    ctrl.workers = args.workers
    ctrl.max_node_wgt = int((5.0 * graph.node_num) / ctrl.coarsen_to)
    ctrl.logger = setup_custom_logger('MILE')

    if ctrl.debug_mode:
        ctrl.logger.setLevel(logging.DEBUG)
    else:
        ctrl.logger.setLevel(logging.INFO)
    ctrl.logger.info(args)


def read_data(ctrl, args):
    prefix = "./dataset/" + args.data
    if args.format == "metis":
        input_graph_path = prefix + ".metis"
        graph, mapping = read_graph(ctrl, input_graph_path, metis=True)
    else:
        input_graph_path = prefix + ".edgelist"
        graph, mapping = read_graph(ctrl, input_graph_path, edgelist=True)

    return input_graph_path, graph, mapping


def select_base_embed(ctrl):
    mod_path = "base_embed_methods." + ctrl.basic_embed
    embed_mod = importlib.import_module(mod_path)
    embed_func = getattr(embed_mod, ctrl.basic_embed)
    return embed_func


def select_refine_model(ctrl):
    refine_model = None
    if ctrl.refine_type == 'MD-gcn':
        refine_model = GCN
    elif ctrl.refine_type == 'MD-gs':
        refine_model = GraphSage
    elif ctrl.refine_type == 'MD-dumb':
        refine_model = GCN
        ctrl.refine_model.untrained_model = True
    return refine_model


def evaluate_embeddings(input_graph_path, mapping, embeddings):
    truth_mat = np.loadtxt(input_graph_path + ".truth").astype(int)  # truth before remapping
    idx_arr = truth_mat[:, 0].reshape(-1)  # this is the original index
    raw_truth = truth_mat[:, 1:]  # multi-class result
    if mapping is not None:
        idx_arr = [mapping.old2new[idx] for idx in idx_arr]
    if args.format == "metis":
        idx_arr = [idx - 1 for idx in idx_arr]  # -1 due to METIS (starts from 1)
    embeddings = embeddings[idx_arr, :]  # in the case of yelp, only a fraction of data contains label.
    truth = raw_truth
    res = eval_multilabel_clf(ctrl, embeddings, truth)
    print res


def store_embeddings(input_graph_path, mapping, embeddings, node_num):
    prefix = "./dataset/" + args.data
    is_metis = (args.format == "metis")
    idx_arr = range(node_num)
    if mapping is not None:
        idx_arr = [mapping.new2old[idx] for idx in idx_arr]
    if is_metis:
        idx_arr = [idx + 1 for idx in idx_arr]  # METIS starts from 1.
    out_file = open(prefix + ".embeddings", "w")
    for i, node_idx in enumerate(idx_arr):
        print >> out_file, str(node_idx) + " " + " ".join(["%.6f" % val for val in embeddings[i]])
    out_file.close()


if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ctrl = Control()
    args = parse_args()

    # Read input graph
    input_graph_path, graph, mapping = read_data(ctrl, args)
    set_control_params(ctrl, args, graph)

    # Coarsen method
    match_method = generate_hybrid_matching

    # Base embedding
    basic_embed = select_base_embed(ctrl)

    # Refinement model
    refine_model = select_refine_model(ctrl)

    # Generate embeddings
    embeddings = multilevel_embed(ctrl, graph, match_method=match_method, basic_embed=basic_embed,
                                  refine_model=refine_model)

    # Evaluate embeddings
    if not args.no_eval:
        evaluate_embeddings(input_graph_path, mapping, embeddings)

    # Store embeddings
    if args.store_embed:
        store_embeddings(input_graph_path, mapping, embeddings, graph.node_num)
