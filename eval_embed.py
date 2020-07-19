from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn import metrics
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.semi_supervised import label_propagation
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import os


def summarize_eval_result(ctrl, metrics_dict):
    res = "************************************************************" + "\n"
    res += "Dataset    :\t" + ctrl.data+ "\n"
    res += "Basic Embed:\t" + ctrl.basic_embed + "\n"
    res += "Refine type:\t" + ctrl.refine_type + "\n"
    res += "Coarsen level:\t" + str(ctrl.coarsen_level) + "\n"

    all_keys = sorted(metrics_dict.keys())
    if 'micro-f1' in all_keys: # particular orders easier to see.
        all_keys = ['micro-f1', 'macro-f1', 'weighted-f1', 'samples-f1']
    for key in all_keys:
        res += key +":\t" + metrics_dict[key] + "\n"
    res += "Consumed time:\t" + "{0:.3f}".format(ctrl.embed_time) + " seconds" + "\n"
    res += "************************************************************" + "\n"
    return res

def eval_multilabel_clf(ctrl, embeddings, truth):
    attributes = embeddings
    ctrl.logger.info("Attributes shape: "+ str(attributes.shape))
    ctrl.logger.info("Truth shape: " + str(truth.shape))
    rnd_time = 10
    test_size = 0.1
    metrics_dict = {'micro': [], 'macro': [], 'weighted': [], 'samples': []}

    for _ in range(rnd_time):
        X_train, X_test, y_train, y_test = train_test_split(attributes, truth, test_size=test_size, random_state=np.random.randint(0, 1000))
        clf = OneVsRestClassifier(LogisticRegression(), n_jobs=12) # for multilabel scenario. #penalty='l2'
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)
        y_pred = []
        for inst in range(len(X_test)):
            # assume it has the same number of labels as the truth. Same strtegy is used in DeepWalk and Node2Vec paper.
            y_pred.append(y_pred_proba[inst, :].argsort()[::-1][:sum(y_test[inst, :])])

        y_pred = MultiLabelBinarizer(range(y_pred_proba.shape[1])).fit_transform(y_pred)
        for key in metrics_dict.keys():
            metrics_dict[key].append(f1_score(y_test, y_pred, average=key))

    return summarize_eval_result(ctrl, {key+'-f1': "{0:.3f}".format(np.mean(metrics_dict[key])) for key in metrics_dict.keys()})

def eval_oneclass_clf(ctrl, embeddings, truth):
    attributes = embeddings
    ctrl.logger.info("Attributes shape: "+ str(attributes.shape))
    ctrl.logger.info("Truth shape: " + str(truth.shape))
    truth = np.argmax(truth, axis=1)
    rnd_time = 10
    test_size = 0.1
    metrics_dict = {'micro': [], 'macro': [], 'weighted': [], 'samples': []}

    for i in range(rnd_time):
        X_train, X_test, y_train, y_test = train_test_split(attributes, truth, test_size=test_size, random_state=i) 
        clf = LogisticRegression(penalty='l2')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        for key in metrics_dict.keys():
            metrics_dict[key].append(f1_score(y_test, y_pred, average=key))

    return summarize_eval_result(ctrl, {key: "{0:.3f}".format(np.mean(metrics_dict[key])) for key in metrics_dict.keys()})


def eval_clustering(ctrl, embeddings, truth):
    '''This is used when evaluating the node embeddings for graph clustering.'''
    cls_alg = KMeans(n_clusters=5000, n_jobs=ctrl.workers) # you might need to change to other clustering algorithm. 
    # The ground truth contains overlapping community.
    pred_labels = cls_alg.fit(embeddings).labels_
    ARI = metrics.adjusted_rand_score(truth, pred_labels)
    FMS = metrics.fowlkes_mallows_score(truth, pred_labels)
    return summarize_eval_result(ctrl, {'ARI': "{0:.3f}".format(ARI), 'FMS': "{0:.3f}".format(FMS)})
