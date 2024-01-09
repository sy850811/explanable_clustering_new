

import csv
import importlib
import json

import numpy as np
import torch

import envf as env

load = importlib.import_module("loadStuff")
root_address = env.root_address
number_of_clusters = env.number_of_clusters
utility = importlib.import_module("utility")
#write the above function using a loop
def concept_vocab():
    unique_concepts = load.unique_concepts_for_each_cluster()
    vocabulary = []
    for i in range(number_of_clusters):
        vocabulary += unique_concepts[i]
    vocabulary = list(set(vocabulary))
    print("Length of vocabulary:",len(vocabulary))

    #store vocabulary in a json file names "vocabulary"
    with open(root_address + env.DATASET_NAME +"/" + "vocabulary.json", "w") as write_file:
        json.dump(vocabulary, write_file)



def compute_distances():
    vocabulary = load.concept_vocabulary()
    cluster_weightDict = load.cluster_weightDictionary()
    distances = np.zeros((len(vocabulary), number_of_clusters),dtype=np.float64)
    for i in range(len(vocabulary)):
        distances[i] = cluster_weightDict.get(vocabulary[i],0)
    # distances[distances == 0] = env.infinite
    # distances[distances == 0] = 0
    with open(root_address + env.DATASET_NAME +"/" + "distances.json", "w") as write_file:
        json.dump(distances.tolist(), write_file)
    return distances

def document_embedding_wrt_concepts_e():
    #we extracted concepts present in each document and return list of 4 list  thus y = cluster number
    importlib.reload(load)
    cluster_wikifier_results = load.wikifier_concepts()
    vocabulary =np.array(load.concept_vocabulary())
    neighbourhood_dictionary = load.neighbourhood_dictionary()
    neighbourhood_sets = [set(neighbourhood_dictionary[concept]) for concept in vocabulary]
    
    cluster_neighbour_distance = load.cluster_neighbour_distance_json()
    # template = [{},{}]
    file_name = 'document_embedding_wrt_concepts.csv'
    document_no = 0
    with open(root_address+env.DATASET_NAME+'/'+file_name, 'w') as f:
        writer = csv.writer(f)
        for y,extracted_document_concepts in enumerate(cluster_wikifier_results):
            ##################remove this line later [:3] is for testing################
            for document in extracted_document_concepts:
                document_no+=1
                doc_embed = [{},{}]
                # doc_neighbourhood = template.copy()
                for i in range(len(vocabulary)):
                    for concept in document:
                        if vocabulary[i] == concept:
                            doc_embed[0][int(i)] = int(doc_embed[0].get(i,0) + 1)# if doc_embed[0] has a key corresponding to i then increment its value by 1 else add a new key value pair with key i and value 1
                            #simply add the tfidf value as the value of the key i
                        # elif doc_embed[0][int(i)] > 0:
                        #     continue
                        elif concept in neighbourhood_sets[i]:
                            #find index of cencept in the neighbourhood_dictionary[concept]
                            # start = time.time()
                            index = neighbourhood_dictionary[vocabulary[i]].index(concept)
                            # end = time.time()
                            # print(end-start)
                            # neighbour_sim = 1 - cluster_neighbour_distance[vocabulary[i]][index]
                            neighbour_sim = cluster_neighbour_distance[vocabulary[i]][index] #bcos used 1/1+euclidean
                            neighbour_sim = 0 if (neighbour_sim<0.2) else neighbour_sim
                            doc_embed[1][int(i)] = doc_embed[1].get(i,0) + neighbour_sim
                            # doc_embed[1][int(i)] = doc_embed[1].get(i,0) + cluster_neighbour_distance[vocabulary[i]][index]
                            # doc_embed[1][int(i)] = doc_embed[1].get(i,0) + 1

                            #figure out the proportion of neighbours of concept i are covered. If the proportion is greater than 0.5 then add the tfidf value as the value of the key i

                doc_embed.append(int(y))
                writer.writerow([doc_embed])             

###updated###
def calculate_neighbour_similarity(cluster_neighbour_distance, concept, index):
    neighbour_sim = 1 - cluster_neighbour_distance[concept][index]
    return neighbour_sim if neighbour_sim >= 0.5 else 0



def process_document(args):
    document, vocabulary, neighbourhood_sets, cluster_neighbour_distance, neighbourhood_dictionary = args
    doc_embed = [{}, {}]

    # for i in range(len(vocabulary)):
    for i in range(len(vocabulary)):
        for concept in document:
            if vocabulary[i] == concept:
                doc_embed[0][int(i)] = int(doc_embed[0].get(i, 0) + 1)
            elif concept in neighbourhood_sets[i]:
                index = neighbourhood_dictionary[vocabulary[i]].index(concept)
                neighbour_sim = calculate_neighbour_similarity(cluster_neighbour_distance, vocabulary[i], index)
                doc_embed[1][i] = doc_embed[1].get(i, 0) + neighbour_sim
    return doc_embed



def save_embeddings_to_csv(embeddings, file_name):
    with open(root_address+env.DATASET_NAME+'/'+file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        for doc_embed in embeddings:
            writer.writerow(doc_embed)

def generator_document_embedding_load(batch_size,filename='document_embedding_wrt_concepts.csv'):
    with open (root_address + env.DATASET_NAME +"/" +filename) as f:
        reader = csv.reader(f)
        data = []
        for line in reader:
            #line is a list of two lists
            concept_list = list(map(int, line[0].strip('[]').split(',')))
            neighbourhood_list = list(map(int, line[1].strip('[]').split(',')))
            # data.append(list(map(int, line[0].strip('[]').split(','))))
            # data.append(list(map(int, line)))
            data.append([concept_list,neighbourhood_list])

            if len(data) == batch_size:
                yield data
                data = []
        if data:
            yield data


#given the size of the embedding and the index and value stored in the dictionary, return a list of zeros with value at the index
def get_list(tensorSize,dict_dict):
    concept_tensor = torch.zeros(tensorSize)
    for key,value in dict_dict.items():
        concept_tensor[key] = value
    return concept_tensor

def embedding_data(tensorSize,list_of_dict):
    y = torch.zeros(len(list_of_dict))
    for index,i in enumerate(list_of_dict):
        if index == 0:
            direct_embedding = get_list(tensorSize,i[0])
            neighbourhood_embedding = get_list(tensorSize,i[1])
        else:
            direct_embedding = torch.cat((direct_embedding,get_list(tensorSize,i[0])),0)
            neighbourhood_embedding = torch.cat((neighbourhood_embedding,get_list(tensorSize,i[1])),0)
        y[index] = i[2]

    return direct_embedding.reshape(len(list_of_dict),-1),neighbourhood_embedding.reshape(len(list_of_dict),-1),y
    
