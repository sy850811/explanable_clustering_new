import importlib
import pickle
import statistics
import torch

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score,normalized_mutual_info_score)

import loadStuff as load



env = importlib.import_module("envf")
root_address = env.root_address
device = env.device

def kmeans():
    print("number of clusters",env.number_of_clusters)
    kmeansModel = None
    bestAcc = 0
    evaluation = [[],[],[]]
    for i in range(10):
        kmeansi,acc,mi,ri = kmeans_algo(random_stat=i)
        evaluation[0].append(acc)
        evaluation[1].append(mi)
        evaluation[2].append(ri)
        if acc > bestAcc:
            bestAcc = acc
            bestMi = mi
            bestRi = ri
            kmeansModel = kmeansi
    print("Best results: ")
    print("acc",bestAcc)
    print("MI", bestMi)
    print("ri",bestRi)
    print("Results Distribution:(Metric:mean,stdev)")
    print("Acc (stdev,mean)",statistics.mean(evaluation[0]),statistics.stdev(evaluation[0]))
    print("MI (stdev,mean)",statistics.mean(evaluation[1]),statistics.stdev(evaluation[1]))
    print("AI (stdev,mean)",statistics.mean(evaluation[2]),statistics.stdev(evaluation[2]))
    return kmeansModel
    
def kmeans_algo(random_stat=1):
    df = load.embeddings()
    df['label'] = df['label'] + 1
    embeddings = torch.tensor(np.array(df['embeddings'].values.tolist()))
    labels = torch.tensor(df['label'].values)
    kmeansModel = KMeans(n_clusters=env.number_of_clusters,random_state=random_stat,n_init=10).fit(embeddings)
    mi = normalized_mutual_info_score(labels, kmeansModel.labels_)
    ri = adjusted_mutual_info_score(labels, kmeansModel.labels_)
    acc = clustering_accuracy(kmeansModel.labels_, labels)
    return kmeansModel,acc,mi,ri


#rewrite above function which works on n clusters
def store_cluster_text_n_centroids(kmeansModel):
    df = load.embeddings()
    clusters = []
    trueLabels = []
    for i in range(kmeansModel.n_clusters):
        clusters.append([])
        trueLabels.append([])
    for i in range(len(kmeansModel.labels_)):
        clusters[kmeansModel.labels_[i]].append(df['text'].iloc[i])
        trueLabels[kmeansModel.labels_[i]].append(df['label'].iloc[i])

    # store all clusters and centroids in a pickle file
    with open(root_address + env.DATASET_NAME +"/" + "clusteredNews.pkl", 'wb') as f:
        pickle.dump([clusters,kmeansModel.cluster_centers_,trueLabels], f)


#given two list (cluster labels and true labels) use Hungarian algorithm to find the best mapping and return the accuracy
def clustering_accuracy(y_pred, y_true):    
    y_true = np.array(y_true,np.int64)
    y_pred = np.array(y_pred,np.int64)
    assert y_pred.size == y_true.size, 'y_pred.size {} y_true.size {}'.format(y_pred.size, y_true.size)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Use linear_sum_assignment from scipy
    rows, cols = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(rows, cols)]) * 1.0 / y_pred.size

