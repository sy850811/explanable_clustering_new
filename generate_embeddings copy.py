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
        concept_support = wikifier.getConceptSupport(r)
        concepts = list(concept_support.keys())
        for concept in concepts:
            support_word_index_list = concept_support[concept]["wordsUsed"]
            # aggregate the embeddings at support word index in batch_document_embedding at index i
            # concept_embedding = np.mean(batch_document_embedding[i][support_word_index_list],axis=0)
            #batch_document_embedding shape -> #Samples,#words,#1024
            concept_embedding = tf.reduce_mean(tf.gather(batch_document_embedding[i], support_word_index_list),axis=0).numpy()
            if concept in concept_embedding_dict:
                # average the embeddings of the concept with the previous embedding
                concept_embedding_dict[concept] = (concept_embedding_dict[concept] + concept_embedding)/2
            else:
                concept_embedding_dict[concept] = concept_embedding
    return concept_embedding_dict


# def get_embeddings(df):
#     embedding = np.array([])
#     index = 0
#     for i in range(0,df.shape[0],10):# what is the size is not multiple of 100
#         mean_embedding = tf.reduce_mean(elmo_model(tf.constant(df["text"][i:i+10]))["elmo"],axis=1).numpy()
#         if index == 0:
#             embedding = mean_embedding
#         else:
#             embedding = np.vstack((embedding,mean_embedding))  
#         index+=1
#     format = []
#     for i in embedding:
#         format.append(i)
#     return format
def get_embeddings(df):
    with open(root_address + env.DATASET_NAME +"/" + "annotation.json", 'r') as f:
        annotatedDocuments = json.load(f)
    concept_embedding_dict= {}
    embedding = np.array([])
    index = 0
    for i in range(0,df.shape[0],10):# what is the size is not multiple of 100
        batch_document_embedding = elmo_model(tf.constant(df["text"][i:i+10]))["elmo"]
        # combine two dictionary
        concept_embedding_dict = {**concept_embedding_dict, **find_concept_embedding(annotatedDocuments,batch_document_embedding,df["text"][i:i+10],concept_embedding_dict)}
        
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
    with open(root_address + env.DATASET_NAME +"/" + "concept_embedding_dict.pkl", 'wb') as f:
        pickle.dump(concept_embedding_dict, f)
    return format_list
     

def generation(df = pd.DataFrame([])):
    if df.empty:
        df = load.test_train_dataset()
    df['embeddings'] = get_embeddings(df)

    # normalize the embeddings
    df['embeddings'] = df['embeddings'].apply(lambda x : normalize([x])[0])

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


# import pandas as pd
# from allennlp.modules.elmo import Elmo, batch_to_ids
# from sklearn.preprocessing import normalize
# import numpy as np
# import loadStuff as load
# import tensorflow as tf

# import envf as env
# device = env.device
# root_address = env.root_address
# counter= 0
# import gc
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# # elmo_model = tf.saved_model.load(
# #     root_address + "models/elmo_3", tags=None, options=None).signatures["default"]

# # encoder = ElmoEmbedder(
# #         options_file='models/elmo_correct/elmo_2x4096_512_2048cnn_2xhighway_options.json',
# #         weight_file='models/elmo_correct/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
# #         cuda_device=0)
# elmo = Elmo(
#     options_file='models/elmo_correct/elmo_2x4096_512_2048cnn_2xhighway_options.json',
#     weight_file='models/elmo_correct/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
#     num_output_representations=1,
#     requires_grad=False,
#     do_layer_norm=False
# )
# #write get_embedding function to generate embeddings using elmo instead of bert

# def get_embeddings(df):

#     batch_size = 10
#     feat_mean = []
#     data_size = len(df.shape[0])
#     sents = df["text"].tolist()


#     embedding = np.array([])
#     for i in range(0,data_size,batch_size):# what is the size is not multiple of 100
#         batch_sents = sents[i: i + batch_size]
#         mean_embedding = tf.reduce_mean(elmo_model(tf.constant(batch_sents))["elmo"],axis=1).numpy()
#         if index == 0:
#             embedding = mean_embedding
#         else:
#             embedding = np.vstack((embedding,mean_embedding))  
#     format = []
#     for i in embedding:
#         format.append(i)
#     return format
    


#     for i in range(0, data_size, batch_size):
#         batch_sents = sents[i: i + batch_size]
#         character_ids = batch_to_ids([s.split() for s in batch_sents]).to(env.device)
#         embeddings = elmo(character_ids)
#         elmo_representations = embeddings['elmo_representations'][0].detach().cpu().numpy()
#         batch_feat_lst = encoder.embed_batch([s.split() for s in batch_sents])
#         feat_mean.extend([np.mean(tmp, axis=0) for tmp in elmo_representations])
#         print(i)
#     return np.stack(feat_mean)
    
# def generation(df = pd.DataFrame([])):
#     if df.empty:
#         df = load.test_train_dataset()
#     df['embeddings'] = get_embeddings(df)

#     # normalize the embeddings
#     # df['embeddings'] = df['embeddings'].apply(lambda x : normalize([x])[0])
#     df['embeddings'] = df['embeddings']/ np.linalg.norm(feat, axis=1, keepdims=True)

#     df.to_pickle(root_address+ env.DATASET_NAME + "/newsTextEmbeddingDF.pkl")

#     return df

# def get_concept_embeddings_using_elmo(strr,reshape = True):
#         #apply elmo model to get embeddings for a string
#         if type(strr) != list:
#             strr = [strr]
#         embedding = tf.reduce_mean(elmo_model(tf.constant(strr))["elmo"],axis=1).numpy()
#         if reshape:
#             embedding = embedding[0].reshape(-1,)
#         return embedding

