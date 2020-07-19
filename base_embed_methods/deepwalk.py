from gensim.models import Word2Vec
import numpy as np
import multiprocessing as mp


# For dynamic loading: name of this method should be the same as name of this python FILE.
def deepwalk(ctrl, graph):
    '''Use DeepWalk as the base embedding method. This is a wrapper method and used by MILE.'''
    args = DeepWalkSetting()
    return DeepWalk_Original(args, embed_dim=ctrl.embed_dim, workers=ctrl.workers, graph=graph, ).get_embeddings()

class DeepWalkSetting:
    '''Configuration parameters for DeepWalk.'''
    def __init__(self):
        self.walk_length = 80
        self.number_walks = 10  # per node
        self.window_size = 10
        self.epoch = 5

class DeepWalk_Original(object):
    '''This is the DeepWalk implementation.'''

    def __init__(self, deep_walk_arguments, embed_dim, graph, workers):

        if graph.node_num > 1e6:  # for large graph, we generate parts of walks each time and keep updating the model.
            iterations = deep_walk_arguments.number_walks
            deep_walk_arguments.number_walks = 1
        else:  # for small graphs, we generate all the paths at once and train the model.
            iterations = 1
        for i in range(iterations):
            all_paths = self.generate_walks(deep_walk_arguments, graph, workers)
            if i == 0:
                word2vec = Word2Vec(sentences=all_paths, min_count=0, size=embed_dim, sg=1, hs=1,
                                    workers=workers,
                                    window=deep_walk_arguments.window_size, iter=deep_walk_arguments.epoch)
            else:
                word2vec.train(all_paths, total_examples=word2vec.corpus_count, epochs=deep_walk_arguments.epoch)
        embeddings = np.zeros((graph.node_num, embed_dim))
        for word in range(graph.node_num):
            embeddings[word] = word2vec[str(word)]

        self.embeddings = embeddings

    def get_embeddings(self):
        return self.embeddings

    def generate_walks(self, deep_walk_arguments, graph, workers):
        def rnd_walk_workers(graph, permuted_idx, proc_begin, proc_end, return_dict):
            ''' workers to generate random walks.'''
            walk_length, window_size = (deep_walk_arguments.walk_length, deep_walk_arguments.window_size)
            all_paths = []
            for _ in range(deep_walk_arguments.number_walks):
                for start_idx in permuted_idx[proc_begin: proc_end]:
                    path = [start_idx]
                    for _ in range(walk_length):
                        curr_idx = path[-1]
                        neigh = graph.get_neighs(curr_idx)
                        wgts = graph.get_neigh_edge_wgts(curr_idx)
                        path.append(np.random.choice(neigh, p=wgts / float(sum(wgts))))
                    all_paths.append(map(str, path))
            return_dict[proc_begin] = all_paths

        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        chunk_size = graph.node_num // workers
        permuted_idx = np.random.permutation(graph.node_num)
        for i in range(workers):
            proc_begin = i * chunk_size
            proc_end = (i + 1) * chunk_size
            if i == workers - 1:
                proc_end = graph.node_num
            p = mp.Process(target=rnd_walk_workers, args=(graph, permuted_idx, proc_begin, proc_end, return_dict))
            jobs.append(p)
        for p in jobs:
            p.start()
        for proc in jobs:
            proc.join()
        all_paths = []
        key_arr = sorted(return_dict.keys())
        np.random.shuffle(key_arr)
        for key in key_arr:
            all_paths += return_dict[key]
        return all_paths
