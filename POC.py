# %% [markdown]
# <center>-----------------------------------Set environment and handle imports-----------------------------------</center>

# %%
import importlib
import os
import time

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import nn

# os.environ['dataset_name_env_var'] = "yahoo" 
# os.environ['split'] = str(0)
# os.environ["cosine"] = "false"
# os.environ["normalize"] = "false"

import argparse
from accelerate import Accelerator
accelerator = Accelerator()
device = 'cuda'
device = accelerator.device
# Argument Parser
parser = argparse.ArgumentParser(description='Choose a Dataset')
parser.add_argument('--dataset', type=str, help='The name of the dataset to be used')
parser.add_argument('--split', type=str, help='The name of the dataset to be used')
parser.add_argument('--cosine', type=str, help='The name of the dataset to be used')
parser.add_argument('--normalize', type=str, help='The name of the dataset to be used')
args = parser.parse_args()

#set environment variable dataset_name_env_var
os.environ["dataset_name_env_var"] = args.dataset 
os.environ["split"] = args.split
os.environ["cosine"] = args.cosine
os.environ["normalize"] = args.normalize
startTimeEntireNotebook = time.time()
utility = importlib.import_module("utility")
env,\
load,\
bert_embeddings,\
clustering,\
wikifier,\
cluster_neighbour,\
concept_embeddings_n_relevance,\
train_help_func,\
baselines,predictionsPowEval\
     = utility.load_modules()

device = env.device

# %%
#load dataset

df,eval_df= load.test_train_dataset()
size_of_dataset = df.shape
print(f"Dataset of size {size_of_dataset} loaded.")
#fix above line
print(df['label'].value_counts())

# %%
# wikifier.datasetWikifier()
# cluster_neighbour.extract_neighbourhood_for_dataset()

# %% [markdown]
# <center>-----------------------------------generate embeddings and normalize-----------------------------------</center>

# %%
print("Gnerating BERT embeddings for the dataset.")
df= bert_embeddings.generation(df)

# %% [markdown]
# <center>-----------------------------------k-Means Clustering-----------------------------------</center>

# %%
print("Clustering the dataset.")
kmeans = clustering.kmeans()
clustering.store_cluster_text_n_centroids(kmeans)

# %% [markdown]
# <center>--------------------------extract entities from clusters--------------------------------</center>

# %%
print("Extracting concepts.")
importlib.reload(wikifier)
wikifier.annotate_cluster_documents()

# %%
wikifier_results = load.wikifier_concepts()
# utility.mostFrequentConcepts() Replace this with truely most relevant concepts

# %% [markdown]
# <center>--------------------------Neighbourhood extraction for concepts--------------------------------</center>

# %%
for i in range(len(wikifier_results)):
    wikifier.get_unique_concepts(wikifier_results[i],i)

# %%
cluster_neighbour.extract_neighbourhood_for_each_cluster()

# %%
clusters_neighbour = load.cluster_neighbour()
for i in range(len(clusters_neighbour)):
    cluster_neighbour.create_dict(clusters_neighbour[i],i)
cluster_neighbour.combine_dictionaries()

# %% [markdown]
# <center>----------------------------extract concept relevance--------------------------------</center>

# %%
unique_concepts = load.unique_concepts_for_each_cluster()
for i in range(len(unique_concepts)):
    concept_embeddings_n_relevance.cluster_embeddings(unique_concepts[i],i) #we are inspecting this
    print("Done for cluster ",i)

# %%
unique_concepts = load.unique_concepts_for_each_cluster()

# %%
wikifier_results = load.wikifier_concepts()

# %%
neighbourhood_dictionary = load.neighbourhood_dictionary() #del

# %%
len(list(neighbourhood_dictionary.keys()))

# %%
neighbourhood_dictionary = load.neighbourhood_dictionary()
concept_embeddings_n_relevance.concept_embedding_neighbour(neighbourhood_dictionary)

# %%
concept_embeddings_n_relevance.overallDict()
cluster_weightDict = load.cluster_weightDictionary()

# %% [markdown]
# <center>-----------------------Check overlap of a document with clusters------------------------</center>

# %%
#check if list in the values of dict,neighbourhood_dictionary is of the same length as in  dict,cluster neighbour distance .json for each key
neighbourhood_dictionary = load.neighbourhood_dictionary()
cluster_neighbour_distance = load.cluster_neighbour_distance_json()
for keys in cluster_neighbour_distance.keys():
    if len(cluster_neighbour_distance[keys]) != len(neighbourhood_dictionary[keys]):
        print("error in cluster ",keys)
        print(len(cluster_neighbour_distance[keys]),len(neighbourhood_dictionary[keys]))
        print("")
        cluster_neighbour_distance[keys] = cluster_neighbour_distance[keys][-len(neighbourhood_dictionary[keys]):]

# %%

train_help_func.concept_vocab()
temp = train_help_func.compute_distances()

# %% [markdown]
# create document embeddings of training data

# %%
train_help_func.document_embedding_wrt_concepts_e()

# %%
clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
trueLabels = [item for sublist in trueLabels for item in sublist]
clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
#true Labels is a list of lists. Convert it into a list
trueLabels = [item for sublist in trueLabels for item in sublist]
batchSize = 4
distances = load.concept_centroid_distances()
distances = torch.tensor(distances).to(device)
vocab_len = len(load.concept_vocabulary())
loss_function = nn.functional.binary_cross_entropy
min_validation_loss = torch.tensor(float('inf')).to(device)

# %% [markdown]
# <!-- #considering both c and n -->
# 
# <!-- # adding semantic value -->
# <!-- accuracy:  0.97
# precision:  0.97
# recall:  0.9716981132075472
# Rand :  0.8448427857772554
# MI 0.7946207065282214 -->
# 

# %%
clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
trueLabels = [item for sublist in trueLabels for item in sublist]
print(set(trueLabels))

# %%
documentEmbedding = load.document_embedding()
datasetSize = len(documentEmbedding)
# similarity = 1 - distances
similarity = distances
#set similarity value to be zero where value is 98
# similarity[similarity == -98] = 0
predictions = torch.tensor([], device=device)
dl_predictions = torch.tensor([], device=device)
for j in range(0,datasetSize,batchSize):

    cx,nx,y = train_help_func.embedding_data(vocab_len,documentEmbedding[j:j+batchSize])
    cx = cx.to(device)
    nx = nx.to(device)
    y = y.to(device)
    dl_predictions = torch.cat((dl_predictions,y),0)
    cy = torch.nn.functional.one_hot(y.type(torch.int64) ,env.number_of_clusters).type(torch.float32)

    # document_concept_score = torch.matmul(cx,similarity) + torch.matmul(nx,similarity)
    document_concept_score = torch.matmul(cx,similarity)

    document_concept_score_dist = torch.nn.functional.softmax(document_concept_score, dim=1)
    prediction = torch.argmax(document_concept_score_dist, dim=1) 

    predictions = torch.cat((predictions,prediction),0)

    #compute loss
    loss = loss_function(document_concept_score_dist,cy)

print("Accuracy-ownapproach",accuracy_score(dl_predictions.cpu(), predictions.cpu()))

# %% [markdown]
# -----------------------------------------------------Results-----------------------------------------------------

# %%
decisionTreeModel,decisionTreePredictions,b_dl_predictions = baselines.decisionTrees()
naiveBayesModel,naiveBayesPredictions,b_dl_predictions = baselines.naiveBayes()
logisticRegressionModel,logisticRegressionPredictions,b_dl_predictions = baselines.logisticRegression()

# %%
baselines.storeResults(dl_predictions,predictions,decisionTreeModel,naiveBayesModel,logisticRegressionModel,decisionTreePredictions,naiveBayesPredictions,logisticRegressionPredictions)

# %%
print("\nAccuracy:")
#print cluster accuracy of dl_predictions and predictions
print("ownapproach",round(accuracy_score(dl_predictions.cpu(), predictions.cpu()),2))
# print only till two decimal
print("naiveBayes",round(accuracy_score(dl_predictions.cpu(), naiveBayesPredictions),2))
print("decisionTree",round(accuracy_score(dl_predictions.cpu(), decisionTreePredictions),2))
print("logisticRegression",round(accuracy_score(dl_predictions.cpu(), logisticRegressionPredictions),2))

print("\nF1:")
#find f1 score of dl_predictions and predictions

print("ownapproach",round(f1_score(dl_predictions.cpu(), predictions.cpu(), average='macro'),2))
print("naiveBayes",round(f1_score(dl_predictions.cpu(), naiveBayesPredictions, average='macro'),2))
print("decisionTree",round(f1_score(dl_predictions.cpu(), decisionTreePredictions, average='macro'),2))
print("logisticRegression",round(f1_score(dl_predictions.cpu(), logisticRegressionPredictions, average='macro'),2))

print("\nPrecision:")
#find precision score of dl_predictions and predictions

print("ownapproach",round(precision_score(dl_predictions.cpu(), predictions.cpu(), average='macro'),2))
print("naiveBayes",round(precision_score(dl_predictions.cpu(), naiveBayesPredictions, average='macro'),2))
print("decisionTree",round(precision_score(dl_predictions.cpu(), decisionTreePredictions, average='macro'),2))
print("logisticRegression",round(precision_score(dl_predictions.cpu(), logisticRegressionPredictions, average='macro'),2))

print("\nRecall:")
#find recall score of dl_predictions and predictions

print("ownapproach",round(recall_score(dl_predictions.cpu(), predictions.cpu(), average='macro'),2))
print("naiveBayes",round(recall_score(dl_predictions.cpu(), naiveBayesPredictions, average='macro'),2))
print("decisionTree",round(recall_score(dl_predictions.cpu(), decisionTreePredictions, average='macro'),2))
print("logisticRegression",round(recall_score(dl_predictions.cpu(), logisticRegressionPredictions, average='macro'),2))

# %%
norm,distanceMetric = True,"cosine"
print(f" norm: {norm}, Distance Metric: {distanceMetric}")
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)
norm,distanceMetric = True,"euclidean"
print(f" norm: {norm}, Distance Metric: {distanceMetric}")
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)
norm,distanceMetric = False,"cosine"
print(f" norm: {norm}, Distance Metric: {distanceMetric}")
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)
norm,distanceMetric = False,"euclidean"
print(f" norm: {norm}, Distance Metric: {distanceMetric}")
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)

# %%
# Ending time
endTimeEntireNotebook = time.time()

# Calculate the total time taken in seconds
total_time_seconds = endTimeEntireNotebook - startTimeEntireNotebook

# Convert the total time taken to hours and minutes
hours, remainder = divmod(total_time_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

# Print the time taken in hours and minutes
print(f"Time taken to run the entire notebook: {int(hours)} hours and {int(minutes)} minutes")

