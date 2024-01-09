import argparse
import importlib
import os



import torch
import torch.nn as nn
from accelerate import Accelerator
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             f1_score, precision_score, recall_score)


accelerator = Accelerator()
device = 'cuda'
device = accelerator.device

# ----------------------------------

# import os
# os.environ['dataset_name_env_var'] = "r2" 
# os.environ['split'] = str(0)
# os.environ["cosine"] = "false"
# os.environ["normalize"] = "true"

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

# ----------------------------------


utility = importlib.import_module("utility")

env,load,bert_embeddings,clustering,wikifier,cluster_neighbour,concept_embeddings_n_relevance,train_help_func,baselines,predictionsPowEval = utility.load_modules()
device = env.device

# ----------------------------------



# ----------------------------------

#load dataset
importlib.reload(load)
df,eval_df= load.test_train_dataset()
size_of_dataset = df.shape
print("size of dataset",size_of_dataset)
df['label'].value_counts()

# ----------------------------------

# importlib.reload(wikifier)
# wikifier.datasetWikifier()
# cluster_neighbour.extract_neighbourhood_for_dataset()

# ----------------------------------

# newsTextEmbeddingDF.pkl file will be generated
importlib.reload(bert_embeddings)
df= bert_embeddings.generation(df)



# ----------------------------------

#split 1
importlib.reload(clustering)
kmeans = clustering.kmeans()
# clustering.kmeans_algo_eval(kmeans)
clustering.store_cluster_text_n_centroids(kmeans)
 


# ----------------------------------

#reload wikifier
importlib.reload(wikifier)
wikifier.annotate_cluster_documents()

# ----------------------------------

wikifier_results = load.wikifier_concepts()

# ----------------------------------



# ----------------------------------

utility.mostFrequentConcepts()

# ----------------------------------

for i in range(len(wikifier_results)):
    wikifier.get_unique_concepts(wikifier_results[i],i)

# ----------------------------------

cluster_neighbour.extract_neighbourhood_for_each_cluster()

# ----------------------------------

importlib.reload(load)
clusters_neighbour = load.cluster_neighbour()
for i in range(len(clusters_neighbour)):
    cluster_neighbour.create_dict(clusters_neighbour[i],i)
cluster_neighbour.combine_dictionaries()

# ----------------------------------

# extract embeddings of concept and store them
importlib.reload(concept_embeddings_n_relevance)
importlib.reload(bert_embeddings)
unique_concepts = load.unique_concepts_for_each_cluster()
for i in range(len(unique_concepts)):
    concept_embeddings_n_relevance.cluster_embeddings(unique_concepts[i],i) #we are inspecting this
    print("Done for cluster ",i)



# ----------------------------------

unique_concepts = load.unique_concepts_for_each_cluster()

# ----------------------------------

wikifier_results = load.wikifier_concepts()

# ----------------------------------

neighbourhood_dictionary = load.neighbourhood_dictionary() #del

# ----------------------------------

len(list(neighbourhood_dictionary.keys()))

# ----------------------------------

#Loop through neighbourhood_dictionary and pass sublist and list index to a function called concept_embedding_neighbour
importlib.reload(concept_embeddings_n_relevance)
importlib.reload(bert_embeddings)
neighbourhood_dictionary = load.neighbourhood_dictionary()
concept_embeddings_n_relevance.concept_embedding_neighbour(neighbourhood_dictionary)
# concept_embeddings_n_relevance.concept_embedding_neighbour_parallel(neighbourhood_dictionary)

# ----------------------------------



# ----------------------------------
 
importlib.reload(concept_embeddings_n_relevance)
importlib.reload(env)
concept_embeddings_n_relevance.overallDict()
cluster_weightDict = load.cluster_weightDictionary()

# ----------------------------------

neighbourhood_dictionary = load.neighbourhood_dictionary()
print(len(list(neighbourhood_dictionary.keys())))

# ----------------------------------

#check if list in the values of dict,neighbourhood_dictionary is of the same length as in  dict,cluster neighbour distance .json for each key
importlib.reload(cluster_neighbour)
importlib.reload(concept_embeddings_n_relevance)
neighbourhood_dictionary = load.neighbourhood_dictionary()
cluster_neighbour_distance = load.cluster_neighbour_distance_json()
for keys in cluster_neighbour_distance.keys():
    if len(cluster_neighbour_distance[keys]) != len(neighbourhood_dictionary[keys]):
        print("error in cluster ",keys)
        print(len(cluster_neighbour_distance[keys]),len(neighbourhood_dictionary[keys]))
        print("")
        cluster_neighbour_distance[keys] = cluster_neighbour_distance[keys][-len(neighbourhood_dictionary[keys]):]



# ----------------------------------

#reload train_help_func module
importlib.reload(train_help_func)
train_help_func.concept_vocab()
temp = train_help_func.compute_distances()

# ----------------------------------



# ----------------------------------



# ----------------------------------

importlib.reload(train_help_func)
importlib.reload(load)
train_help_func.document_embedding_wrt_concepts_e()
#last run for r2 took time: 1 minute

# ----------------------------------

# import csv
# maxInt = sys.maxsize
# import ast
# while True:
#     # decrease the maxInt value by factor 10 
#     # as long as the OverflowError occurs.

#     try:
#         csv.field_size_limit(maxInt)
#         break
#     except OverflowError:
#         maxInt = int(maxInt/10)
# document_embedding_5 = []
# with open('document_embedding_wrt_conceptsThres0_5.csv') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         # document_embedding.append(ast.literal_eval(row[0]))
#         refined = ast.literal_eval(row[0])
#         refinedint = [{},{},refined[2]]
#         #refined is a list with 3 items: dictionary, dictionary, variable
#         #write code to convert the keys and values of dictionary as smallest integer type

#         #convert keys and values of dictionary as smallest integer type

#         for i in range(2):
#             for key in refined[i].keys():
#                 refinedint[i][int(key)] = refined[i].get(key)
#         #check size of refined and refinedint
#         # print(sys.getsizeof(refined))
#         # print(sys.getsizeof(refinedint))
#         document_embedding_5.append(refinedint)

# ----------------------------------

clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
#true Labels is a list of lists. Convert it into a list
# print(trueLabels)
trueLabels = [item for sublist in trueLabels for item in sublist]
# print(trueLabels)

load = importlib.import_module("loadStuff")

train_help_func = importlib.import_module("train_help_func")

env = importlib.import_module("envf")

#load true_labels

importlib.reload(load)
clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
#true Labels is a list of lists. Convert it into a list
trueLabels = [item for sublist in trueLabels for item in sublist]


device = env.device


batchSize = 4
distances = load.concept_centroid_distances()
distances = torch.tensor(distances).to(device)

importlib.reload(load)

importlib.reload(train_help_func)
# 4:25


vocab_len = len(load.concept_vocabulary())


loss_function = nn.functional.binary_cross_entropy
# initialize adam optimizer 


min_validation_loss = torch.tensor(float('inf')).to(device)


# ----------------------------------



# ----------------------------------



# ----------------------------------



# ----------------------------------

clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
trueLabels = [item for sublist in trueLabels for item in sublist]
print(set(trueLabels))

# ----------------------------------

documentEmbedding = load.document_embedding()
datasetSize = len(documentEmbedding)

# documentEmbedding = sklearn.utils.shuffle(documentEmbedding, random_state=1)
# trueLabels = sklearn.utils.shuffle(trueLabels, random_state=1)

# import numpy as np
# neighbourhood_dictionary = load.neighbourhood_dictionary()
# vocabulary =np.array(load.concept_vocabulary())
# list_of_neighbourhood_values = []
# neighbourhood_sets = [set(neighbourhood_dictionary[concept]) for concept in vocabulary]
# for document in documentEmbedding:
#     for key,value in document[1].items():
#         document[1][key] = value/len(neighbourhood_sets[key])
#         list_of_neighbourhood_values.append(value/len(neighbourhood_sets[key]))



# similarity = 1 - distances
similarity = distances #sorry for naming wrong
#set similarity value to be zero where value is 98
similarity[similarity == -98] = 0
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

# ----------------------------------



# ----------------------------------

importlib.reload(baselines)
decisionTreeModel,decisionTreePredictions,b_dl_predictions = baselines.decisionTrees()
naiveBayesModel,naiveBayesPredictions,b_dl_predictions = baselines.naiveBayes()
logisticRegressionModel,logisticRegressionPredictions,b_dl_predictions = baselines.logisticRegression()

# ----------------------------------

"""expected parameters
pseudo_trueLables, predictions, decisionTree, naiveBayes, logisticRegression,decisionTreePredictions,naiveBayesPredictions,logisticRegressionPredictions"""

baselines.storeResults(dl_predictions,predictions,decisionTreeModel,naiveBayesModel,logisticRegressionModel,decisionTreePredictions,naiveBayesPredictions,logisticRegressionPredictions)

# ----------------------------------



# ----------------------------------

# def removeFunctionAndUncommentCodeIfNeeded()
    # accuracy = accuracy_score(dl_predictions.cpu(), predictions.cpu())
    # print("accuracy: ",accuracy)

    # precision = precision_score(dl_predictions.cpu(), predictions.cpu(), average='macro')
    # print("precision: ",precision)

    # recall = recall_score(dl_predictions.cpu(), predictions.cpu(), average='macro')
    # print("recall: ",recall)

    # print("accuracy",clustering.clustering_accuracy(predictions.cpu(),trueLabels))
    # #evaluate the unsupervised clustering with adjusted rand score
    # print("Rand : ",adjusted_rand_score(trueLabels, predictions.cpu()))
    # #evaluate the unsupervised clustering with adjusted mutual information
    # print("MI",adjusted_mutual_info_score(trueLabels, predictions.cpu()))

    # print("accuracy",clustering.clustering_accuracy(predictions.cpu(),dl_predictions.cpu()))
    # #evaluate the unsupervised clustering with adjusted rand score
    # print("Rand : ",adjusted_rand_score(trueLabels, dl_predictions.cpu()))
    # #evaluate the unsupervised clustering with adjusted mutual information
    # print("MI",adjusted_mutual_info_score(trueLabels, dl_predictions.cpu()))

    # print("accuracy",clustering.clustering_accuracy(dl_predictions.cpu(),predictions.cpu()))
    # #evaluate the unsupervised clustering with adjusted rand score
    # print("Rand : ",adjusted_rand_score(np.array(dl_predictions.cpu()), predictions.cpu()))
    # #evaluate the unsupervised clustering with adjusted mutual information
    # print("MI",adjusted_mutual_info_score(np.array(dl_predictions.cpu()), predictions.cpu()))

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

# ----------------------------------



# ----------------------------------

importlib.reload(predictionsPowEval)
importlib.reload(load)
norm,distanceMetric = True,"cosine"
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)
norm,distanceMetric = True,"euclidean"
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)
norm,distanceMetric = False,"cosine"
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)
norm,distanceMetric = False,"euclidean"
predictionsPowEval.storeConceptEmbeddingTestingData(predictionsPowEval.annotate_unseen_documents(eval_df),norm)
predictionsPowEval.unseen_predictions(distanceMetric)

# ----------------------------------

#reload 
importlib.reload(predictionsPowEval)
importlib.reload(load)


# ----------------------------------



# ----------------------------------



# ----------------------------------



# ----------------------------------



# ----------------------------------



# ----------------------------------



# ----------------------------------

#agnews
"""
done with annotations
accuracy
proposed 0.8175
decisionTree 0.5825
naiveBayes 0.675
logisticRegression 0.69
f1
proposed 0.812097350223977
decisionTree 0.5713470155464406
naiveBayes 0.6737946435601784
logisticRegression 0.6785735283187455
precision
proposed 0.8202423432563014
decisionTree 0.6215235537295283
naiveBayes 0.6931779850844599
logisticRegression 0.7007342310177362
recall
proposed 0.8131957450669443
decisionTree 0.5716630509212249
naiveBayes 0.6725769897462739
logisticRegression 0.6773570241120668
"""
#r2
"""
accuracy 
proposed 0.78839590443686
decisionTree 0.7679180887372014
naiveBayes 0.8071672354948806
logisticRegression 0.8481228668941979
f1
proposed 0.7704139020537124
decisionTree 0.7673870150490935
naiveBayes 0.7997199260798423
logisticRegression 0.8473006643302387
precision
proposed 0.8350639977611831
decisionTree 0.7682800396062671
naiveBayes 0.8181990881458967
logisticRegression 0.8465336134453781
recall
proposed 0.7677900858161956
decisionTree 0.7711097246583244
naiveBayes 0.7957656947109442
logisticRegression 0.849515591707966
"""
#r5
"""
accuracy
proposed 0.6005873715124816
decisionTree 0.5770925110132159
naiveBayes 0.6916299559471366
logisticRegression 0.6666666666666666
f1
proposed 0.5854369435919631
decisionTree 0.5462249699716223
naiveBayes 0.6522856670485132
logisticRegression 0.6323269552253373
precision
proposed 0.6472839349012866
decisionTree 0.5702396680717686
naiveBayes 0.7129414814433599
logisticRegression 0.69326630753066
recall
proposed 0.6038787829349865
decisionTree 0.5427833001215822
naiveBayes 0.6445923888765883
logisticRegression 0.6157772909310555
"""
#yahoo
"""
done with annotations
accuracy
proposed 0.227
decisionTree 0.336
naiveBayes 0.403
logisticRegression 0.442
f1
proposed 0.2521681266010469
decisionTree 0.34459212497084657
naiveBayes 0.44894058107538404
logisticRegression 0.46656041270054305
precision
proposed 0.19467456357250526
decisionTree 0.38995208144803895
naiveBayes 0.42549318759315524
logisticRegression 0.5291226583727322
recall
proposed 0.45274066450790584
decisionTree 0.33012243905853145
naiveBayes 0.5127904458397454
logisticRegression 0.437272370749288
"""

# ----------------------------------

# def removeFunctionAndUncommentCodeIfNeeded()
    # importlib.reload(train_help_func)
    # train_help_func.document_embedding_wrt_concepts_e()
    # clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
    # #true Labels is a list of lists. Convert it into a list
    # # print(trueLabels)
    # trueLabels = [item for sublist in trueLabels for item in sublist]
    # # print(trueLabels)
    # from sklearn.metrics import adjusted_mutual_info_score
    # from sklearn.metrics.cluster import adjusted_rand_score
    # import torch
    # import torch.nn as nn
    # import sklearn
    # import importlib
    # load = importlib.import_module("loadStuff")

    # train_help_func = importlib.import_module("train_help_func")

    # env = importlib.import_module("envf")

    # #load true_labels

    # importlib.reload(load)
    # clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
    # #true Labels is a list of lists. Convert it into a list
    # trueLabels = [item for sublist in trueLabels for item in sublist]


    # device = env.device

    # import time
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import precision_score
    # from sklearn.metrics import recall_score
    # datasetSize = 100
    # batchSize = 10
    # distances = load.concept_centroid_distances()
    # distances = torch.tensor(distances).to(device)

    # importlib.reload(load)

    # importlib.reload(train_help_func)
    # # 4:25


    # vocab_len = len(load.concept_vocabulary())


    # loss_function = nn.functional.binary_cross_entropy
    # # initialize adam optimizer 


    # min_validation_loss = torch.tensor(float('inf')).to(device)
    # documentEmbedding = load.document_embedding()
    # documentEmbedding = sklearn.utils.shuffle(documentEmbedding, random_state=1)
    # trueLabels = sklearn.utils.shuffle(trueLabels, random_state=1)

    # # import numpy as np
    # # neighbourhood_dictionary = load.neighbourhood_dictionary()
    # # vocabulary =np.array(load.concept_vocabulary())
    # # list_of_neighbourhood_values = []
    # # neighbourhood_sets = [set(neighbourhood_dictionary[concept]) for concept in vocabulary]
    # # for document in documentEmbedding:
    # #     for key,value in document[1].items():
    # #         document[1][key] = value/len(neighbourhood_sets[key])
    # #         list_of_neighbourhood_values.append(value/len(neighbourhood_sets[key]))



    # similarity = 1 - distances
    # #set similarity value to be zero where value is 98
    # similarity[similarity == -98] = 0
    # predictions = torch.tensor([], device=device)
    # dl_predictions = torch.tensor([], device=device)

    # for j in range(0,datasetSize,batchSize):

    #     cx,nx,y = train_help_func.embedding_data(vocab_len,documentEmbedding[j:j+batchSize])
    #     cx = cx.to(device)
    #     nx = nx.to(device)
    #     y = y.to(device)
    #     dl_predictions = torch.cat((dl_predictions,y),0)

    #     cy = torch.nn.functional.one_hot(y.type(torch.int64) ,env.number_of_clusters).type(torch.float32)


    #     document_concept_score = torch.matmul(cx,similarity) + torch.matmul(nx,similarity)
    #     # document_concept_score = torch.matmul(nx,similarity)

    #     document_concept_score_dist = torch.nn.functional.softmax(document_concept_score, dim=1)
    #     prediction = torch.argmax(document_concept_score_dist, dim=1) 

    #     predictions = torch.cat((predictions,prediction),0)

    #     #compute loss
    #     loss = loss_function(document_concept_score_dist,cy)

    # accuracy = accuracy_score(dl_predictions.cpu(), predictions.cpu())
    # print("accuracy: ",accuracy)

    # precision = precision_score(dl_predictions.cpu(), predictions.cpu(), average='macro')
    # print("precision: ",precision)

    # recall = recall_score(dl_predictions.cpu(), predictions.cpu(), average='macro')
    # print("recall: ",recall)

    # #evaluate the unsupervised clustering with adjusted rand score
    # print("Rand : ",adjusted_rand_score(trueLabels, predictions.cpu()))
    # #evaluate the unsupervised clustering with adjusted mutual information
    # print("MI",adjusted_mutual_info_score(trueLabels, predictions.cpu()))

# ----------------------------------

# def removeFunctionAndUncommentCodeIfNeeded()
    # import torch
    # import torch.nn as nn
    # import os
    # import sklearn
    # import importlib
    # load = importlib.import_module("loadStuff")

    # train_help_func = importlib.import_module("train_help_func")

    # env = importlib.import_module("envf")

    # device = env.device

    # import time
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import precision_score
    # from sklearn.metrics import recall_score
    # #set datasetSize = no_of_samples from environment variable
    # datasetSize = 10
    # batchSize = 10
    # distances = load.concept_centroid_distances()
    # distances = torch.tensor(distances).to(device)

    # importlib.reload(load)

    # importlib.reload(train_help_func)
    # # 4:25


    # vocab_len = len(load.concept_vocabulary())

    # # alpha = torch.tensor[0](1, requires_grad=True, device=device)#*0 + 0.8512
    # alpha = torch.tensor([3.5], requires_grad=True, device=device)

    # beta = torch.tensor([3.5], requires_grad=True, device=device)

    # gamma = torch.tensor([3.5], requires_grad=True, device=device)

    # alpha_n = torch.tensor([3.5], requires_grad=True, device=device)

    # beta_n = torch.tensor([3.5], requires_grad=True, device=device)

    # gamma_n = torch.tensor([3.5], requires_grad=True, device=device)

    # #initialize alpha and beta with value 0.8512, 0.2242, 0.0723, 
    # loss_function = nn.functional.binary_cross_entropy
    # # initialize adam optimizer 
    # optimizer = torch.optim.Adam([alpha,beta,gamma,alpha_n,beta_n,gamma_n], lr=0.3)


    # min_validation_loss = torch.tensor(float('inf')).to(device)
    # documentEmbedding = load.document_embedding()
    # documentEmbedding = sklearn.utils.shuffle(documentEmbedding)
    # import numpy as np
    # neighbourhood_dictionary = load.neighbourhood_dictionary()
    # vocabulary =np.array(load.concept_vocabulary())
    # list_of_neighbourhood_values = []
    # neighbourhood_sets = [set(neighbourhood_dictionary[concept]) for concept in vocabulary]
    # for document in documentEmbedding:
    #     for key,value in document[1].items():
    #         document[1][key] = value/len(neighbourhood_sets[key])
    #         list_of_neighbourhood_values.append(value/len(neighbourhood_sets[key]))


    # optimizer = torch.optim.Adam([alpha,beta,gamma,alpha_n,beta_n,gamma_n], lr=0.003)

    # alpha_list = []
    # beta_list = []
    # alpha_n_list = []
    # beta_n_list = []
    # loss_list = []
    # accuracy_list = []
    # precision_list = []
    # recall_list = []
    # for i in range(100):
    #     predictions = torch.tensor([], device=device)
    #     dl_predictions = torch.tensor([], device=device)

        
    #     batch_loss_list = []
    #     for j in range(0,datasetSize,batchSize):
    #         # check if float value is nan 
    #         # if i == 49 and j == 974:
    #         #     pass
    #         # if torch.isnan(torch.tensor(alpha.view(-1).item())):
    #         #     pass
    #         cx,nx,y = train_help_func.embedding_data(vocab_len,documentEmbedding[j:j+batchSize])

    #         cx = cx.to(device)

    #         nx = nx.to(device)

    #         y = y.to(device)
    #         dl_predictions = torch.cat((dl_predictions,y),0)

    #         cy = torch.nn.functional.one_hot(y.type(torch.int64) ,env.number_of_clusters).type(torch.float32)

    #         document_concept_score = torch.matmul(cx,torch.exp(-(alpha * distances) + beta)) + torch.matmul(nx,torch.exp(-(alpha_n * distances) + beta_n))

    #         # document_concept_score = torch.matmul(nx,torch.exp(-(alpha_n * distances) + beta_n))

    #         #use formula gamma/(beta + alpha * distance)
    #         # document_concept_score = torch.matmul(cx,(gamma/(beta**2 + alpha**2 * distances))) #+ torch.matmul(nx,(gamma_n**2/(beta_n**2 + alpha_n**2 * distances)))

    #         document_concept_score_dist = torch.nn.functional.softmax(document_concept_score, dim=1)
    #         prediction = torch.argmax(document_concept_score_dist, dim=1) 

    #         predictions = torch.cat((predictions,prediction),0)

    #         #compute loss
    #         loss = loss_function(document_concept_score_dist,cy)
    #         batch_loss_list.append(loss.view(-1).item())
    #         #compute gradient
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         # print(j)
    #         if j % 1000 == 0:
    #             alpha_list.append(alpha.view(-1).item())
    #             beta_list.append(beta.view(-1).item())
    #             alpha_n_list.append(alpha_n.view(-1).item())
    #             beta_n_list.append(beta_n.view(-1).item())

    #     loss_list.append(sum(batch_loss_list)/len(batch_loss_list))
    #     accuracy = accuracy_score(dl_predictions.cpu(), predictions.cpu())
    #     print("accuracy: ",accuracy)
    #     print("loss of batch : ",sum(batch_loss_list)/len(batch_loss_list))

    #     precision = precision_score(dl_predictions.cpu(), predictions.cpu(), average='macro')
    #     # print("precision: ",precision)

    #     recall = recall_score(dl_predictions.cpu(), predictions.cpu(), average='macro')
    #     # print("recall: ",recall)
    #     print("alpha",alpha.view(-1).item(),"\t","alpha_n",alpha_n.view(-1).item(),"\epoch = ",i)
    #     accuracy_list.append(accuracy)
    #     precision_list.append(precision)
    #     recall_list.append(recall)

    #     #evaluate the unsupervised clustering with adjusted rand score
    #     print("Rand : ",adjusted_rand_score(trueLabels, predictions.cpu()))
    #     #evaluate the unsupervised clustering with adjusted mutual information
    #     print("MI",adjusted_mutual_info_score(trueLabels, predictions.cpu()))

    # #plot a graph to show the progression of alpha and beta, alpha_n and beta_n
    # import matplotlib.pyplot as plt
    # plt.plot(alpha_list)
    # plt.plot(beta_list)
    # plt.plot(alpha_n_list)
    # plt.plot(beta_n_list)
    # plt.plot(loss_list)
    # plt.legend(["alpha","beta","alpha_n","beta_n","loss"])
    # #show index as epoch number on x axis
    # plt.show()

    # # plot a graph showing loss
    # plt.plot(loss_list)
    # plt.legend(["loss"])
    # plt.xlabel("epoch")
    # plt.show()

    # # plot a graph showing accuracy, precision and recall
    # plt.plot(accuracy_list)
    # plt.plot(precision_list)
    # plt.plot(recall_list)
    # plt.legend(["accuracy","precision","recall"])
    # plt.xlabel("epoch")
    # plt.show()

# ----------------------------------



# ----------------------------------



# ----------------------------------
