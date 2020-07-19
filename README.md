### **MILE: A Multi-Level Framework for Scalable Graph Embedding**
MILE is a multi-level framework to scale up existing graph embedding techniques, without modifying them. 
It incorporates existing embedding techniques as black boxes, and can improves the scalability of 
many embedding methods by reducing both the running time and memory consumption.
Additionally, MILE also provides a lift in the quality of node embeddings in most of the cases.
Read [our paper](https://arxiv.org/pdf/1802.09612.pdf) for more details. Here we publish our code and data with a brief instruction on how to add a new embedding method as the base embedding technique.

#### **Required Packages**
* tensorflow
* numpy
* scipy
* scikit-learn
* networkx
* gensim (only for using DeepWalk as base embedding method)
* theano (only for using NetMF as base embedding method)

#### **Input and Output**
* Input graph: For now, we only support undirected graph. The input graph can be weighted or unweighted. Two different formats are supported: edgelist and metis.
  - Edgelist: The input file must contain ".edgelist" extension. Each of line is an edge with an optional number for weight, separated by a space: *node1 node2 weight*. To save space, each edge should only show up once in the file with node2 as the one with larger id, i.e., node1 <= node2. If the graph is unweighted, *weight* can be left as empty. The node index does not need to be continuous as an internal node mapping will be conducted if necessary. See `./dataset/PPI.edgelist` for an example.
  - Metis: This input format has been commonly used for the METIS package. A detailed definition of the format can be found [here](http://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html). A graph of N nodes is stored in a file of N+1 lines. The first line contains two required numbers: the number of nodes and the number of edges, and an optional number to denote weighted or unweighted graph: 0 or missing means unweighted while 1 means weighted. Each subsequent line lists the neighbors and edges weight (if any) of a node. Note that this format requires the index to be continuous starting from **1**. See `./dataset/PPI.metis` for an example.

* Input ground truth for evaluation: If the evaluation is enabled (without `--no-eval`), a file proving ground truth for multi-class classification is required. It contains the list of nodes with known labels, where each line corresponds to one node. For each line, the first number is the index of the node (should be the same as the input graph) and the remaining numbers form a vector representation of the label (multi-class allowed). See `./dataset/PPI.edgelist.truth` (corresponding to `PPI.edgelist`) or `./dataset/PPI.metis.truth` (corresponding to `PPI.metis`) for examples.

* Output embeddings: if `--store-embed` is enabled, the embeddings will be saved under the `./dataset/` directory with the extension of `.embeddings`. Each row corresponds to one node, where the first number is the original node id and the rest is the vector representation.

#### **How To Run**
Use `python main.py` to run the code with all the default settings. Here are some useful arguments that can be passed to the program:
* `--data`: name of the dataset (file located in `./dataset/`), e.g., `--data PPI`.
* `--format`: the format of a dataset, should be either *edgelist* or *metis*, e.g., `--format metis` (by default it is *metis*).
* `--basic-embed`: name of the base embedding method, e.g., `--basic-embed deepwalk`.
* `--coarsen-level`: number of levels for coarsening, e.g., `--coarsen-level 2`.
* `--embed-dim`: dimensionality for embedding, e.g., `--embed-dim 128`.
* `--store-embed`: will store the embeddings if enabled.
* `--no-eval`: will not evaluate the embeddings if set (.truth file will not be required then).
* `--workers`: number of processes to run the code. 
* `--refine-type`: refinement method, including `MD-gcn` (the one proposed in MILE), `MD-dumb` (without model training), and `MD-gs` (using GraphSAGE).


#### **Adding a New Base Embedding Method**
Follow the steps below to add a new base embedding method (say `DeepWalk`):
  1. Create a python file in `./base_embed_methods/` (e.g., `deepwalk.py`).
  2. Include the original embedding implementation in that file (e.g., `def DeepWalk_Original(...)` in `deepwalk.py`).
  3. Provide a wrapper method to generate embeddings using the original embedding method; this wrapper will be called by MILE framework (e.g., `def deepwalk(...)` in `deepwalk.py`).<br/><b>NOTE</b>: The wrapper method should be same as the filename in Step 1.
  4. Add the name of the wrapper method in Step 3 as a choice to `--basic-embed` in the arguments of `./main.py`.

#### **Citation**
If you use any part of our code, please cite our work:

J. Liang, S. Gurukar, S. Parthasarathy. "MILE: A Multi-Level Framework for Scalable Graph Embedding". *arXiv preprint arXiv:1802.09612*, 2018. \[[PDF](https://arxiv.org/pdf/1802.09612.pdf)\]\[[bib](publications/MILE.txt)\]
