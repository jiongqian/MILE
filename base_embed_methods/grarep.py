from numpy import linalg as la
from sklearn.preprocessing import normalize
from utils import graph_to_adj
import math
import numpy as np
import scipy.sparse as sp


# For dynamic loading: name of this method should be the same as name of this python FILE.
def grarep(ctrl, graph):
    '''Use GraRep as the base embedding method. This is a wrapper method and used by MILE.'''
    args = GraRepSetting()
    return GraRep_Original(graph, args.Kstep, ctrl.embed_dim, logger=ctrl.logger).get_embeddings()


class GraRepSetting:
    '''Configuration parameters for GraRep.'''
    def __init__(self):
        self.Kstep = 4  # Note: embed_dim % Kstep  == 0

class GraRep_Original(object):
    '''This is the original implementation of GraRep. Code is adapted from https://github.com/thunlp/OpenNE.'''

    def __init__(self, graph, Kstep, dim, logger=None):
        self.logger = logger
        self.graph = graph
        self.Kstep = Kstep
        self.node_size = graph.node_num
        assert dim % Kstep == 0  # dim = 128, Kstep = 4 (usually 3 to 6)
        self.dim = dim / Kstep
        self.train()

    def getAdjMat(self):
        adj = graph_to_adj(self.graph)
        return normalize(adj, norm='l1', axis=1)  # rescale it row-wise.

    def GetProbTranMat(self, Ak):
        Ak = Ak.tocoo()
        rows = Ak.row
        cols = Ak.col
        data = Ak.data
        col_sum = [0.0] * self.node_size
        for idx in range(len(cols)):
            col_sum[cols[idx]] += data[idx]
        new_rows = []
        new_cols = []
        new_data = []
        for idx in range(len(rows)):
            data[idx] = np.log(data[idx] / col_sum[cols[idx]]) - np.log(1.0 / self.node_size)
            if data[idx] > 0:
                new_data.append(data[idx])
                new_rows.append(rows[idx])
                new_cols.append(cols[idx])
        return sp.csr_matrix((new_data, (new_rows, new_cols)), shape=(self.node_size, self.node_size))

    def GetRepUseSVD(self, probTranMat, alpha):
        U, S, VT = sp.linalg.svds(probTranMat, k=min(self.dim, self.node_size - 1))
        Ud = U[:, 0:self.dim]
        Sd = S[0:self.dim]
        return np.array(Ud) * np.power(Sd, alpha).reshape((self.dim))

    def train(self):
        adj = self.getAdjMat()
        node_size = adj.shape[0]
        Ak = sp.csr_matrix(adj)
        self.RepMat = np.zeros((self.node_size, self.dim * self.Kstep))
        for i in range(self.Kstep):
            self.logger.info('Kstep = ' + str(i))
            if i > 0:
                Ak = Ak.dot(adj)
            probTranMat = self.GetProbTranMat(Ak)
            Rk = self.GetRepUseSVD(probTranMat, 0.5)
            Rk = normalize(Rk, axis=1, norm='l2')
            self.RepMat[:, self.dim * i:self.dim * (i + 1)] = Rk[:, :]
        # get embeddings
        self.vectors = self.RepMat

    def get_embeddings(self):
        return self.vectors
