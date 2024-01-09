
import ast
import csv
import importlib
import json
import os
import pickle
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

env = importlib.import_module("envf")
wikifier = importlib.import_module("cluster_wikifier")
number_of_clusters = env.number_of_clusters
root_address = env.root_address


def concept_neighbour_distance_dict():
    with open(root_address + env.DATASET_NAME +"/" + "cluster_neighbour_distance.json", "r") as read_file:
        concept_neighbour_distance_dictionary = json.load(read_file)
    return concept_neighbour_distance_dictionary
    

def embeddings():
    return pd.read_pickle(root_address + env.DATASET_NAME +"/" + "newsTextEmbeddingDF.pkl")

 
def clusters_n_centroids_n_labels():

    with open(root_address + env.DATASET_NAME +"/"+ "clusteredNews.pkl", 'rb') as f:
        clusters, centroids,trueLabels = pickle.load(f)

    return clusters, centroids,trueLabels
    

def wikifier_concepts():
    wikifier_results = []
    for i in range(number_of_clusters):
        with open(root_address + env.DATASET_NAME +"/" + 'training_data_wikification'+str(i)+'.json', 'r') as f:
            wikifier_results.append(json.load(f))
    return wikifier_results



def unique_concepts_for_each_cluster():
    unique_concepts = []
    for i in range(number_of_clusters):
        with open(root_address + env.DATASET_NAME +"/" + 'cluster'+str(i)+'_unique_concepts.json', 'r') as f:
            unique_concepts.append(json.load(f))
    return unique_concepts


def unique_concept_embeddings_for_each_cluster():
    cluster_concepts_embeddings_df = []
    for i in range(number_of_clusters):
        cluster_concepts_embeddings_df.append(pd.read_pickle(root_address + env.DATASET_NAME +"/" + "cluster"+str(i)+"_concepts_embeddings_df.pkl"))
    return cluster_concepts_embeddings_df


def cluster_neighbour():
    clusters_neighbour = []
    for i in range(number_of_clusters):
        with open(root_address + env.DATASET_NAME +"/" + "cluster"+str(i)+"_neighbour.json", "r") as read_file:
            clusters_neighbour.append(json.load(read_file))
    return clusters_neighbour


def cluster_neighbourhood_dictionaries():
    neighbourhood_dictionaries = []
    for i in range(number_of_clusters):
        with open(root_address + env.DATASET_NAME +"/" + "neighbourhood_"+str(i)+"_dictionary.json", "r") as read_file:
            neighbourhood_dictionaries.append(json.load(read_file))
    return neighbourhood_dictionaries


def extracted_concepts():
    with open(root_address + env.DATASET_NAME +"/" + "extracted_document_concepts0.json", 'r') as f:
        extracted_document_concepts0 = json.load(f)

    with open(root_address + env.DATASET_NAME +"/" + "extracted_document_concepts1.json", 'r') as f:

        extracted_document_concepts1 = json.load(f)

    with open(root_address + env.DATASET_NAME +"/" + "extracted_document_concepts2.json", 'r') as f:
        extracted_document_concepts2 = json.load(f)


    with open(root_address + env.DATASET_NAME +"/" + "extracted_document_concepts3.json", 'r') as f:
        extracted_document_concepts3 = json.load(f)

    return extracted_document_concepts0, extracted_document_concepts1, extracted_document_concepts2, extracted_document_concepts3


def cluster_weightDictionary():
    with open(root_address + env.DATASET_NAME +"/" + 'cluster_weightDict.json') as json_file:
        cluster_weightDict = json.load(json_file)

    return cluster_weightDict

    


def locally_fetch_dataset_concepts():
    filename = os.path.join(root_address +env.DATASET_NAME+ "/annotation.json")
    # Ensure the file exists before trying to open it
    with open(filename, 'r') as file:
        data = json.load(file)
        
        # Initialize a list to hold all the strings within the lists in all the values
        dataset_concepts = []
        
        # Iterate over the dictionary and extract all strings from the lists in the values
        for value in data.values():
            value = wikifier.extract_titles(value)#contexual embedding because now annotation contains raw json earlier it had already extracted titles. but now we are getting them at this point.
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        dataset_concepts.append(item)

    return list(set(dataset_concepts))


def agnews():
    columnNames = ['label','skip','text']
    df = pd.read_csv(root_address + env.DATASET_NAME +"/" + "train.csv",names = columnNames,skiprows=1)
    df.drop(columns=['skip'],inplace=True)
    df['label'] = df['label'].apply(lambda x : x -1)
    return df

def dbpedia():
    columnNames = ['label','skip','text']
    df = pd.read_csv(root_address + env.DATASET_NAME +"/" + "train.csv",names = columnNames)
    df.drop(columns=['skip'],inplace=True)
    df['label'] = df['label'].apply(lambda x : x -1)
    return df

def r2():
    columnNames = ['text','label']
    df = pd.read_csv(root_address + env.DATASET_NAME +"/" + "r2.csv",names = columnNames,skiprows=1)
    print(df.head())
    return df 

def r5():
    columnNames = ['text','label']
    df = pd.read_csv(root_address + env.DATASET_NAME +"/" + "r5.csv",names = columnNames,skiprows=1)
    return df 

def yahoo():
    df = pd.read_csv(root_address + "yahoo" +"/" + "train.csv")
    return df


def test_train_dataset():
    columnNames = ['label','text']
    df = pd.read_csv(root_address + env.DATASET_NAME +"/" + "train.csv",names= columnNames)
    # if env.device == "mps":
        # df = df.groupby('label').head(5)
    #split into train and test with ratio 90:10
    train, test = train_test_split(df, test_size=0.1, random_state=env.random_state)
    return train,test




def concept_vocabulary():
    with open(root_address + env.DATASET_NAME +"/" + "vocabulary.json", "r") as read_file:
        vocabulary = json.load(read_file)
    return vocabulary

def concept_centroid_distances():
    with open(root_address + env.DATASET_NAME +"/" + "distances.json", "r") as read_file:
        distances = json.load(read_file)

    return distances

#write function to load a pickel file named neighbourhood_dictionary.pkl
def neighbourhood_dictionary():
    with open(root_address + env.DATASET_NAME +"/" + "neighbourhood_dictionary.pkl", 'rb') as f:
        neighbourhood_dictionary_list = pickle.load(f)
    return neighbourhood_dictionary_list
#load cluster_neighbour_distance.json its a single file in which dictionary was stored
def cluster_neighbour_distance_json():
    with open(root_address + env.DATASET_NAME +"/" + "cluster_neighbour_distance.json", "r") as read_file:
        cluster_neighbour_distance = json.load(read_file)

    return cluster_neighbour_distance


def document_embedding():
    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)
    document_embedding_list = []
    with open(root_address + env.DATASET_NAME +"/" + 'document_embedding_wrt_concepts.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            refined = ast.literal_eval(row[0])
            refinedint = [{},{},refined[2]]
            #refined is a list with 3 items: dictionary, dictionary, variable
            for i in range(2):
                for key in refined[i].keys():
                    refinedint[i][int(key)] = refined[i].get(key)
            #check size of refined and refinedint
            # print(sys.getsizeof(refined))
            # print(sys.getsizeof(refinedint))
            document_embedding_list.append(refinedint)
    return document_embedding_list


def tfIdf():
    #load the vectors from the file
    with open('tfidf_vectors.csv') as f:
        reader = csv.reader(f)
        vectors = []
        for line in reader:
            vectors.append(list(map(float, line)))
    return vectors

def Baselines():
    #load model and prediction for each random state
    baselines_models = {'naiveBayes':[],'decisionTrees':[],'randomForest':[]}
    baselines_predictions = {'naiveBayes':[],'decisionTrees':[],'randomForest':[]}
    for baseline in env.baselines:
        model_path = root_address + env.DATASET_NAME +"/" + baseline+str(random_state)+"_model.sav"
        prediction_path = root_address + env.DATASET_NAME +"/" + baseline+str(random_state)+"_prediction.sav"
        
        for random_state in env.random_states:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(prediction_path, 'rb') as prediction_file:
                prediction = pickle.load(prediction_file)

            baselines_models[baseline].append(model)
            baselines_predictions[baseline].append(prediction)
            
    return baselines_models,baselines_predictions


def baselineModels():
    prefix = env.root_address + env.DATASET_NAME +"/"+env.DATASET_NAME + str(env.random_state)
    
    with open(prefix+ "decisionTree.pkl", 'rb') as file:
        decisionTreeModel = pickle.load(file)
    
    with open(prefix + "naiveBayes.pkl", 'rb') as file:
        naiveBayesModel = pickle.load(file)
    
    with open(prefix + "logisticRegression.pkl", 'rb') as file:
        logisticRegressionModel = pickle.load(file)
    
    return decisionTreeModel, naiveBayesModel, logisticRegressionModel

def concept_embedding_dict():
    with open(root_address + env.DATASET_NAME +"/" + "concept_embedding_dict.pkl", 'rb') as f:
        concept_embedding_dict_pkl = pickle.load(f)
    return concept_embedding_dict_pkl

    #write function to load concept_embedding_dict_test.pkl
def concept_embedding_dict_test():
    with open(root_address + env.DATASET_NAME +"/" + "concept_embedding_dict_test.pkl", 'rb') as f:
        concept_embedding_dict_test_pkl = pickle.load(f)
    return concept_embedding_dict_test_pkl

def userStudyData(noOfDocuments=env.userStudyDocumentsCount):
    _,eval_df = test_train_dataset()
    #fetch 20 random documents from the eval_df with random seed 42
    userStudy_df = eval_df.sample(n=noOfDocuments,random_state=42).reset_index(drop=True)
    return userStudy_df