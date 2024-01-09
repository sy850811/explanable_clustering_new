#generate embeddings
import importlib
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize

import envf as env
import loadStuff as load

wikifier = importlib.import_module("cluster_wikifier")

device = env.device
root_address = env.root_address
counter= 0
# import gc

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

elmo_model = tf.saved_model.load(
    root_address + "models/elmo_3", tags=None, options=None).signatures["default"]

def find_concept_embedding(annotatedDocuments,batch_document_embedding,batch_document_text,concept_embedding_dict):
    """
    Given:
    batch of document Text: concepts in the document 
    Embedding of documents batch 
    Find:
    Dictionary of concepts and their embeddings
    Solution:
    -) Loop through each document
    -) Extract the concepts from the document
    -) extract the words index of the cocnepts
    -)print the word index from both the annotation.json and the document in batch_document -text
    -) Loop through each concept
    -) Get the embedding of the concept
    -) find the mean of the embedding of multi worded concepts
    -) store the concept and its embedding in a dictionary
    """
    for i,document in enumerate(batch_document_text):
        r = annotatedDocuments[document]
        document = document.replace("\n"," ").replace("\t"," ")
        concept_support = wikifier.getConceptSupport(r,document)
        concepts = list(concept_support.keys())
        for concept in concepts:
            support_word_index_list = concept_support[concept]["wordsUsed"]
            support_word_index_list_int = tf.cast(support_word_index_list, tf.int32)
            concept_embedding = tf.reduce_mean(tf.gather(batch_document_embedding[i], support_word_index_list_int),axis=0).numpy().astype(np.float64)

            #set numpy datatype to np.float64

            if concept in concept_embedding_dict:
                current_average = concept_embedding_dict[concept][0]
                current_frequency = concept_embedding_dict[concept][1]

                new_frequency = current_frequency + 1
                new_value = concept_embedding
                new_average = current_average + (new_value - current_average)/float(new_frequency)
                concept_embedding_dict[concept] = [new_average,new_frequency]
            else:
                concept_embedding_dict[concept] = [concept_embedding,float(1)]
    return concept_embedding_dict

def join_dict(dict1, dict2):
    for key in dict2:
        if key in dict1:
            # Ensure that both elements are numpy arrays for safe addition
            freq1 = dict1[key][1]
            freq2 = dict2[key][1]
            avg1 = np.array(dict1[key][0], dtype=np.float64)
            avg2 = np.array(dict2[key][0], dtype=np.float64)
            combined_freq = freq1 + freq2
            weight = freq2 / combined_freq
            combined_avg = avg1 + weight * (avg2 - avg1)
            
            # Update the dictionary
            dict1[key] = [combined_avg, combined_freq]
        else:
            # If the key is not in dict1, add it directly
            dict1[key] = dict2[key]
    return dict1

def get_embeddings(df,train):

    if train:
        filename = "concept_embedding_dict.pkl"
    else:
        filename = "concept_embedding_dict_test.pkl"
    with open(root_address + env.DATASET_NAME +"/" + "annotation.json", 'r') as f:
        annotatedDocuments = json.load(f)
    concept_embedding_dict= {}
    embedding = np.array([])
    index = 0
    batchSize = 10
    for i in range(0,df.shape[0],batchSize):
        batch = df["text"][i:i+batchSize].apply(lambda x: x.replace("\n"," ").replace("\t"," "))
        batch_document_embedding = elmo_model(tf.constant(batch))["elmo"]
        find_concept_embedding(annotatedDocuments,batch_document_embedding,df["text"][i:i+batchSize],concept_embedding_dict)
        # concept_embedding_dict = join_dict(concept_embedding_dict,newDict)
        mean_embedding = tf.reduce_mean(batch_document_embedding,axis=1).numpy()

        if index == 0: 
            embedding = mean_embedding
        else:
            embedding = np.vstack((embedding,mean_embedding))  
        index+=1
    format_list = []
    for i in embedding:
        format_list.append(i)
    #store the concept_embedding_dict
    with open(root_address + env.DATASET_NAME +"/" + filename, 'wb') as f:
        pickle.dump(concept_embedding_dict, f)
    return format_list
     

def generation(df = pd.DataFrame([]),train=True):
    if df.empty:
        df = load.test_train_dataset()
    df['embeddings'] = get_embeddings(df,train)

    # normalize the embeddings
    df['embeddings'] = df['embeddings'].apply(lambda x : normalize([x])[0])
    if train:
        df.to_pickle(root_address+ env.DATASET_NAME + "/newsTextEmbeddingDF.pkl")
    return df

def get_concept_embeddings_using_elmo(strr,reshape = True):
    #apply elmo model to get embeddings for a string
    if type(strr) != list:
        strr = [strr]
    embedding = tf.reduce_mean(elmo_model(tf.constant(strr))["elmo"],axis=1).numpy()
    # # if env.tmp_normalize == True:
    # if False:
    #     embedding = normalize(embedding)
    if reshape:
        embedding = embedding[0].reshape(-1,)
    return embedding


