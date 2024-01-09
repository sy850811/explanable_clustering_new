"""
After models are trained here we generate predictions for the unseen data
"""
import importlib
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from generate_embeddings import elmo_model
from scipy.spatial.distance import cosine as cosine_distance
from scipy.spatial.distance import euclidean as euclidean_distance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import normalize

# import warnings
# warnings.filterwarnings('error')


load = importlib.import_module("loadStuff")
env = importlib.import_module("envf")
root_address = env.root_address
wikifier = importlib.import_module("cluster_wikifier")
bert_embeddings = importlib.import_module("generate_embeddings")
device = env.device
root_address = env.root_address
# elmo_model = tf.saved_model.load(
#     root_address + "models/elmo_3", tags=None, options=None).signatures["default"]
def unseen_predictions(distanceMetric): 
    """
    #load the unseen documents
    #preprocess the unseen documents
    #kmeans prediction
    #explanation model predictions:
        #baselines
        #proposed model
    #evaluate the predictions of explanation models(proposed,DT,NB,LR) against kmeans predictions
    """
    _,eval_df = load.test_train_dataset()
    annotated_unseen_documents = annotate_unseen_documents(eval_df)
    kmeans_predictions = get_kmeans_predictions(eval_df)
    decisionTreeModel,naiveBayesModel,logisticRegressionModel = load.baselineModels()
    baseline_preprocessed_data = []
    for doc in annotated_unseen_documents:
        #concat the elements of the list with a space to form a string
        for concept in doc:
            concept = concept.replace("_"," ")
        baseline_preprocessed_data.append(" ".join(doc))
        

    decisionTreePredictions = decisionTreeModel.predict(baseline_preprocessed_data)
    naiveBayesPredictions = naiveBayesModel.predict(baseline_preprocessed_data)
    logisticRegressionPredictions = logisticRegressionModel.predict(baseline_preprocessed_data)
    _,ourOwnPredictions,_ = ownPredictions(eval_df,distanceMetric) 
    performance(kmeans_predictions,decisionTreePredictions,naiveBayesPredictions,logisticRegressionPredictions,ourOwnPredictions)

def annotate_unseen_documents(eval_df):
    """
    Given: text documents
    Find: concepts in the text documents
    solution: From annotation.json, get concepts corresponding to the text documents
    Format:
    document_concepts = [[c1,c2,c3], #doc1
                        [c1,c2,c3],  #doc2
                        [c1,c2,c3]   #doc3
    """

    filename = os.path.join(root_address +env.DATASET_NAME+ "/annotation.json")
    # Ensure the file exists before trying to open it
    with open(filename, 'r') as file:
        annotatedDocuments = json.load(file)
    documents = eval_df["text"].tolist()
    document_concepts = []
    for doc in documents:
        # document_concepts.append(annotatedDocuments[doc])
        document_concepts.append(wikifier.extract_titles(annotatedDocuments[doc]))
    filename = os.path.join(root_address +env.DATASET_NAME+ "/document_concepts.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(document_concepts, f)
    print("done with annotations")
    return document_concepts

def get_kmeans_predictions(eval_df):
    """
    #generate embeddings for the unseen documents
    #get the cluster centers
    #get the kmeans predictions based on distance of embedding and clkuster centers
    #fromat [custer index for doc1,
            cluster index for doc2,
            cluster index for doc3]
    """
    
    eval_df = bert_embeddings.generation(eval_df,train=False) #already normalized
    _, centroids,_ = load.clusters_n_centroids_n_labels() #clusters, centroids,trueLabels
    kmeans_predictions = []
    for i in range(len(eval_df)):
        concept_embeddings = eval_df["embeddings"].iloc[i]
        kmeans_predictions.append(get_nearest_centroid(concept_embeddings,centroids))

    return kmeans_predictions

def get_nearest_centroid(embedding,centroids):
    """
    Given:
    1) list of embeddings corresponding to list of concepts of 1 document
    2) list of cluster centers
    Find:
    1) distance of each embedding from each cluster center
    2) return the index of the cluster center with minimum distance
    Format: [cluster indexfor concept]
    """
    distances = []
    for centroid in centroids:
        # if env.tmp_cosineDistance:
        # if True:
        #     distances.append(cosine_distance(embedding,centroid))
        # else:
        distances.append(euclidean_distance(embedding,centroid))
    return distances.index(min(distances))
    
def performance(truePredictions,decisionTreePredictions,naiveBayesPredictions,logisticRegressionPredictions,ourOwnPredictions):
    """
    Given:
    1) true labels
    2) predictions of each model
    Find:
    1) accuracy
    2) f1
    3) precision
    4) recall
    """

    #accuracy
    print("accuracy")
    print("proposed",accuracy_score(truePredictions,ourOwnPredictions))
    print("decisionTree",accuracy_score(truePredictions,decisionTreePredictions))
    print("naiveBayes",accuracy_score(truePredictions,naiveBayesPredictions))
    print("logisticRegression",accuracy_score(truePredictions,logisticRegressionPredictions))
    #f1
    print("f1")
    print("proposed",f1_score(truePredictions,ourOwnPredictions,average='macro'))
    print("decisionTree",f1_score(truePredictions,decisionTreePredictions,average='macro'))
    print("naiveBayes",f1_score(truePredictions,naiveBayesPredictions,average='macro'))
    print("logisticRegression",f1_score(truePredictions,logisticRegressionPredictions,average='macro'))

    #precision
    print("precision")
    print("proposed",precision_score(truePredictions,ourOwnPredictions,average='macro'))
    print("decisionTree",precision_score(truePredictions,decisionTreePredictions,average='macro'))
    print("naiveBayes",precision_score(truePredictions,naiveBayesPredictions,average='macro'))
    print("logisticRegression",precision_score(truePredictions,logisticRegressionPredictions,average='macro'))
    #recall
    print("recall")
    print("proposed",recall_score(truePredictions,ourOwnPredictions,average='macro'))
    print("decisionTree",recall_score(truePredictions,decisionTreePredictions,average='macro'))
    print("naiveBayes",recall_score(truePredictions,naiveBayesPredictions,average='macro'))
    print("logisticRegression",recall_score(truePredictions,logisticRegressionPredictions,average='macro'))

def storeConceptEmbeddingTestingData(annotated_unseen_documents,norm):
    filename = os.path.join(root_address +env.DATASET_NAME+ "/concept_embeddings_testing_data.json")
    df_concept_embeddings = pd.DataFrame(columns=["concept","embedding"])
    df_concept_embeddings = {}
    for doc in annotated_unseen_documents:
        for concept in doc:
            concept_embedding = bert_embeddings.get_concept_embeddings_using_elmo(concept)
            if norm:
                concept_embedding = normalize(concept_embedding.reshape(1,-1)).reshape(-1)
            df_concept_embeddings[concept] = concept_embedding.tolist()
    with open(filename, 'w') as f:
        json.dump(df_concept_embeddings, f)


def ownPredictions(eval_df,distanceMetric):

    # Provide:
    # 1) document:
    #                         dataframe with relevant column:'text'
    documents = eval_df["text"].tolist()
    
    # 2) annotations
    #                         # Format: [[c1,c2,c3], #doc1
    #                         #         [c1,c2,c3],  #doc2
    #                         #         [c1,c2,c3]   #doc3
    
    with open(root_address + env.DATASET_NAME +"/" + "annotation.json", 'r') as f:
        annotatedDocuments = json.load(f)

    
    # 3) centroids
                            #print an check. I think they are list of embeddings
    
    _,centroids,_ = load.clusters_n_centroids_n_labels()
    
    #also we will store the document,its concepts, concept's similarity to each centroid
    explanation_details = {}

    # To Find:
    document_assignment_scores = [0 for i in range(len(centroids))]
    document_predictions = [None for i in range(len(documents))]
    for doc_no,document in enumerate(documents): 
        r = annotatedDocuments[document]
        #replace tabs and line breaks with space
        document = document.replace("\n"," ").replace("\t"," ")
        # get concept embedding
        document_embedding = elmo_model(tf.constant([document]))["elmo"][0]

        concept_support = wikifier.getConceptSupport(r,document)
        concepts = list(concept_support.keys())
        explanation_details[document] = {}
        document_assignment_scores = [0 for i in range(len(centroids))]# format [0,0]
        for concept in concepts:
            explanation_details[document][concept] = []
            support_word_index_list = concept_support[concept]["wordsUsed"]
            #convert support word index list trype int16
            support_word_index_list_int = tf.cast(support_word_index_list, tf.int32)
            concept_embedding = tf.reduce_mean(tf.gather(document_embedding, support_word_index_list_int),axis=0).numpy().astype(np.float64).reshape(-1,)
            #calculate assignmentscore based on similarity of each concept with each centroid
            if concept in ["Stock","Mergers and acquisitions","Stock","Electric charg","Corporation"]:
                pass
            for index,centroid in enumerate(centroids):
                if distanceMetric =="cosine":
                    fi = 1 - cosine_distance(concept_embedding,centroid.reshape(-1,))
                    document_assignment_scores[index]+= fi
                else:
                    fi = 1/(1+ euclidean_distance(concept_embedding,centroid.reshape(-1,)))
                    document_assignment_scores[index]+= fi
                explanation_details[document][concept].append(fi)
        document_predictions[doc_no] = document_assignment_scores.index(max(document_assignment_scores))
    return (document_assignment_scores,document_predictions,explanation_details)
