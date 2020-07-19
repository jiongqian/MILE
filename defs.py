import numpy as np
import tensorflow as tf

# A Control instance stores most of the configuration information.
class Control:
    def __init__(self):
        self.data = None
        self.workers = 4
        self.coarsen_to = 500  # 1000
        self.coarsen_level = 0  #
        self.max_node_wgt = 100  # to avoid super-node being too large.
        self.embed_dim = 128
        self.basic_embed = "DEEPWALK"
        self.refine_type = "MD-gcn"
        self.refine_model = RefineModelSetting()
        self.embed_time = 0.0  # keep track of the amount of time spent for embedding.
        self.debug_mode = False  # set to false for time measurement.
        self.logger = None


class RefineModelSetting:
    def __init__(self):
        self.double_base = False
        self.learning_rate = 0.001
        self.epoch = 200
        self.early_stopping = 50  # Tolerance for early stopping (# of epochs).
        self.wgt_decay = 5e-4
        self.regularized = True
        self.hidden_layer_num = 2
        self.act_func = tf.tanh
        self.tf_optimizer = tf.train.AdamOptimizer
        self.lda = 0.05  # self-loop weight lambda
        self.untrained_model = False  # if set. The model will be untrained.

        # The following ones are for GraphSAGE only.
        self.gs_sample_neighbrs_num = 100
        self.gs_mlp_layer = 2
        self.gs_concat = True
        self.gs_uniform_sample = False
        self.gs_self_wt = True
