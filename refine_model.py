import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from utils import graph_to_adj


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def diag_ones(shape, name=None):
    """All ones."""
    initial = tf.diag(np.ones(shape[0], dtype=np.float32))
    return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_to_gcn_adj(adj, lda):  # D^{-0.5} * A * D^{-0.5} : normalized, symmetric convolution operator.
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    self_loop_wgt = np.array(adj.sum(1)).flatten() * lda  # self loop weight as much as sum. This is part is flexible.
    adj_normalized = normalize_adj(adj + sp.diags(self_loop_wgt))
    return adj_normalized


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


class GCN:
    '''Refinement model based on the Graph Convolutional Network model (GCN). 
    Parts of the code are adapted from https://github.com/tkipf/gcn'''

    def __init__(self, ctrl, session):
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.args = ctrl.refine_model
        self.session = session
        self.model_dict = dict()
        self.build_tf_graph()

    def build_tf_graph(self):
        act_func = self.args.act_func
        wgt_decay = self.args.wgt_decay
        regularized = self.args.regularized
        learning_rate = self.args.learning_rate
        hidden_layer_num = self.args.hidden_layer_num
        tf_ops = self.args.tf_optimizer
        vars_arr = []

        # placeholders
        input_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="input_embed")  # None: node_num
        gcn_A = tf.sparse_placeholder(tf.float32)  # node_num * node_num
        expected_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="expected_embed")

        curr = input_embed

        for i in range(hidden_layer_num):
            W_hidd = glorot((self.embed_dim, self.embed_dim), name="W_" + str(i))
            curr = act_func(dot(dot(gcn_A, curr, sparse=True), W_hidd))
            vars_arr.append(W_hidd)

        pred_embed = tf.nn.l2_normalize(curr, axis=1)  # this normalization is necessary.
        loss = 0.0
        loss += tf.losses.mean_squared_error(expected_embed, pred_embed) * self.embed_dim

        if regularized:
            for var in vars_arr:
                loss += tf.nn.l2_loss(var) * wgt_decay

        optimizer = tf_ops(learning_rate=learning_rate).minimize(loss)

        self.model_dict['input_embed'] = input_embed
        self.model_dict['gcn_A'] = gcn_A
        self.model_dict['expected_embed'] = expected_embed
        self.model_dict['pred_embed'] = pred_embed
        self.model_dict['optimizer'] = optimizer
        self.model_dict['loss'] = loss
        self.model_dict['vars_arr'] = vars_arr
        init = tf.global_variables_initializer()
        self.session.run(init)

    def train_model(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        '''Train the refinement model.'''
        if self.args.untrained_model:  # this is the 'MD-dumb' model, which will not train the model.
            return
        normalized_A = preprocess_to_gcn_adj(graph_to_adj(fine_graph), self.args.lda)
        gcn_A = sparse_to_tuple(normalized_A)

        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)  # projected embedings.
        else:
            initial_embed = fine_embed

        early_stopping = self.args.early_stopping
        self.logger.info("initial_embed: " + str(initial_embed.shape))
        self.logger.info("fine_embed: " + str(fine_embed.shape))
        loss_arr = []
        self.logger.info("Refinement Model Traning: ")
        for i in range(self.args.epoch):
            optimizer, loss = self.session.run([self.model_dict['optimizer'], self.model_dict['loss']], feed_dict={
                self.model_dict['input_embed']: initial_embed,
                self.model_dict['gcn_A']: gcn_A,
                self.model_dict['expected_embed']: fine_embed})
            loss_arr.append(loss)
            if i % 20 == 0:
                self.logger.info("  GCN iterations-" + str(i) + ": " + str(loss))
            if i > early_stopping and loss_arr[-1] > np.mean(loss_arr[-(early_stopping + 1):-1]):
                self.logger.info("Early stopping...")
                break

    def refine_embedding(self, coarse_graph=None, fine_graph=None, coarse_embed=None):
        '''Apply the learned model for embeddings refinement.'''
        normalized_A = preprocess_to_gcn_adj(graph_to_adj(fine_graph), self.args.lda)
        gcn_A = sparse_to_tuple(normalized_A)
        initial_embed = fine_graph.C.dot(coarse_embed)
        refined_embed, = self.session.run([self.model_dict['pred_embed']], feed_dict={
            self.model_dict['input_embed']: initial_embed,
            self.model_dict['gcn_A']: gcn_A})
        return refined_embed


def alias_setup(probs):
    '''Compute utility lists for non-uniform sampling from discrete distributions. For GraphSAGE only.'''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''Draw sample from a non-uniform discrete distribution using alias sampling. For GraphSAGE only.'''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class GraphSage:
    '''Refinement model based on the GraphSage. Parts of the code are adapted from https://github.com/williamleif/GraphSAGE.'''

    def __init__(self, ctrl, session):
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.num_neighbors = ctrl.refine_model.gs_sample_neighbrs_num
        self.args = ctrl.refine_model
        self.session = session
        self.model_dict = dict()
        self.mlp_layer = ctrl.refine_model.gs_mlp_layer
        self.gs_concat = ctrl.refine_model.gs_concat
        self.uniform_sample = ctrl.refine_model.gs_uniform_sample
        self.self_wt = ctrl.refine_model.gs_self_wt
        self.build_tf_graph()

    def build_tf_graph(self):
        act_func = self.args.act_func
        wgt_decay = self.args.wgt_decay
        regularized = self.args.regularized
        learning_rate = self.args.learning_rate
        hidden_layer_num = self.args.hidden_layer_num
        mlp_layer = self.mlp_layer  # before max-pooling
        mlp_dim = self.embed_dim
        num_neighbors = self.num_neighbors
        concat = self.gs_concat
        tf_ops = self.args.tf_optimizer
        vars_arr = []

        # placeholders
        input_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="input_embed")  # None: node_num
        neigh_vecs = tf.placeholder(tf.float32, shape=[None, self.embed_dim],
                                    name="neigh_vecs")  # n*neigh_sample, embed_dim

        expected_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="expected_embed")

        prev_dim = self.embed_dim
        self_vecs = input_embed
        curr = neigh_vecs
        for i in range(mlp_layer):
            mlp_weights = glorot((prev_dim, mlp_dim), name="mlp_weights_" + str(i))
            mlp_bias = zeros([mlp_dim], name="mlp_bias_" + str(i))
            curr = tf.nn.relu(dot(curr, mlp_weights) + mlp_bias)
            vars_arr.append(mlp_bias)
            vars_arr.append(mlp_weights)
            prev_dim = mlp_dim

        neigh_reshape = tf.reshape(curr, (-1, num_neighbors, mlp_dim))
        max_pool = tf.reduce_max(neigh_reshape, axis=1)

        neigh_wgts = glorot((mlp_dim, self.embed_dim), name="neigh_wgts")
        self_wgts = glorot((self.embed_dim, self.embed_dim), name="self_wgts")
        from_neighs = tf.matmul(max_pool, neigh_wgts)
        from_self = tf.matmul(self_vecs, self_wgts)
        if not concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output_1 = tf.concat([from_self, from_neighs], axis=1)
            weights_compress = glorot((2 * self.embed_dim, self.embed_dim), name="W_concat")
            weight_bias = zeros([self.embed_dim], name="mlp_bias_" + str(i))
            output = dot(output_1, weights_compress) + weight_bias
            vars_arr.append(weight_bias)
            vars_arr.append(weights_compress)

        output = act_func(output)

        pred_embed = tf.nn.l2_normalize(output, axis=1)
        loss = 0.0
        loss += tf.reduce_mean(tf.reduce_sum(tf.square(expected_embed - pred_embed), axis=1))

        optimizer = tf_ops(learning_rate=learning_rate).minimize(loss)

        self.model_dict['input_embed'] = input_embed
        self.model_dict['neigh_vecs'] = neigh_vecs
        self.model_dict['expected_embed'] = expected_embed
        self.model_dict['pred_embed'] = pred_embed
        self.model_dict['optimizer'] = optimizer
        self.model_dict['loss'] = loss
        self.model_dict['vars_arr'] = vars_arr
        init = tf.global_variables_initializer()
        self.session.run(init)

    def sample_from_adj(self, adj, sample_size):
        adj_sample = np.zeros((adj.shape[0], sample_size), dtype=np.int32)
        for i in range(adj.shape[0]):
            row = np.squeeze(adj.getrow(i).toarray())
            if not self.self_wt:
                row[i] = 0
            sampled_row = np.nonzero(row)
            if not self.uniform_sample:
                weights = np.ravel(row[sampled_row])
                dist = weights / weights.sum()
                J, q = alias_setup(dist)
                for j in range(sample_size):
                    adj_sample[i, j] = sampled_row[0][alias_draw(J, q)]
            else:
                sample = np.random.randint(sampled_row[0].size, size=sample_size)  # with replacement.
                adj_sample[i, :] = np.take(sampled_row[0], sample)

        return adj_sample.flatten()

    def train_model(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)
        else:
            initial_embed = fine_embed

        neigh_list = self.sample_from_adj(graph_to_adj(fine_graph), self.num_neighbors)
        neigh_vecs = initial_embed[neigh_list]

        early_stopping = self.args.early_stopping
        self.logger.info("initial_embed: " + str(initial_embed.shape))
        self.logger.info("fine_embed: " + str(fine_embed.shape))
        loss_arr = []
        for i in range(self.args.epoch):
            optimizer, loss = self.session.run([self.model_dict['optimizer'], self.model_dict['loss']], feed_dict={
                self.model_dict['input_embed']: initial_embed,
                self.model_dict['neigh_vecs']: neigh_vecs,
                self.model_dict['expected_embed']: fine_embed})
            loss_arr.append(loss)
            if i % 20 == 0:
                self.logger.info("  GraphSAGE iterations-" + str(i) + ": " + str(loss))
            if i > early_stopping and loss_arr[-1] > np.mean(loss_arr[-(early_stopping + 1):-1]):
                self.logger.info("Early stopping...")
                break

    def refine_embedding(self, coarse_graph=None, fine_graph=None, coarse_embed=None):
        normalized_A = preprocess_to_gcn_adj(graph_to_adj(fine_graph), self.args.lda)
        initial_embed = fine_graph.C.dot(coarse_embed)

        neigh_list = self.sample_from_adj(graph_to_adj(fine_graph), self.num_neighbors)
        neigh_vecs = initial_embed[neigh_list]

        refined_embed, = self.session.run([self.model_dict['pred_embed']], feed_dict={
            self.model_dict['input_embed']: initial_embed,
            self.model_dict['neigh_vecs']: neigh_vecs})
        return refined_embed
