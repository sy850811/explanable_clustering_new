import importlib
import json

bert_embeddings = importlib.import_module("generate_embeddings")
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean as euclidean_distance

load = importlib.import_module("loadStuff")
# sequential_wikifier = importlib.import_module("cluster_wikifier_sequential")
env = importlib.import_module("envf")
root_address = env.root_address
from scipy.spatial.distance import cosine as cosine_distance

# def removeFunctionAndUncommentCodeIfNeeded()
    # class CompletedCounter:
    #     def __init__(self):
    #         self._count = 0
    #         self._lock = threading.Lock()

    #     def increment(self):
    #         with self._lock:
    #             self._count += 1
    #             if self._count % 100 ==0:
    #                 print(f"Completed concepts: {self._count}")

    # import concurrent.futures

    # def calculate_distance(concept, neighbour, concept_neighbour_distance):
    #     concept_embedding = bert_embeddings.get_concept_embeddings_using_elmo(concept)
    #     neighbour_embedding = bert_embeddings.get_concept_embeddings_using_elmo(neighbour)
    #     distance = cosine_distance(concept_embedding, neighbour_embedding)
    #     return distance

    # def process_neighbours(concept, neighbours, concept_neighbour_distance,completed_counter):

    #     for neighbour in neighbours:
    #         distance = calculate_distance(concept, neighbour, concept_neighbour_distance)
    #         concept_neighbour_distance[concept] = concept_neighbour_distance.get(concept, []) + [distance]
    #     completed_counter.increment()
# def concept_embedding_neighbour_parallel(concept_neighbour_dict, endName="cluster_neighbour_distance.json"):
    #     concept_neighbour_distance = {}
    #     print(len(list(concept_neighbour_dict.keys())))
    #     completed_counter = CompletedCounter()
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # You can use ThreadPoolExecutor or ProcessPoolExecutor
    #         futures = []
    #         for concept, neighbours in concept_neighbour_dict.items():
    #             futures.append(executor.submit(process_neighbours, concept, neighbours, concept_neighbour_distance,completed_counter))
            
    #         # Wait for all futures to complete
    #         concurrent.futures.wait(futures)

    #     # Save concept_neighbour_distance to endName (you'll need to implement this)
    #     with open(root_address + endName, "w") as write_file:
    #         json.dump(concept_neighbour_distance, write_file)
    #     return concept_neighbour_distance



def concept_embedding_neighbour(concept_neighbour_dict, endName="cluster_neighbour_distance.json"):

    # try:
    #     concept_neighbour_distance = load.concept_neighbour_distance_dict()
    #     index = len(list(concept_neighbour_distance.keys()))
    # except:
    # print("File not found")
    concept_neighbour_distance = {}
    index = 0


    for concept in list(concept_neighbour_dict.keys()):
        # if index%100 == 0 and index>0:
        #     print(index)
        #     with open(root_address + env.DATASET_NAME +"/" + endName, "w") as write_file:
        #         json.dump(concept_neighbour_distance, write_file)

        concept_embedding = bert_embeddings.get_concept_embeddings_using_elmo(concept)
        
        for neighbour in concept_neighbour_dict[concept]:

            neighbour_embedding = bert_embeddings.get_concept_embeddings_using_elmo(neighbour)
            
            # distance = 1/(1+cosine_distance(concept_embedding,neighbour_embedding))
            distance = 1/(1+euclidean_distance(concept_embedding,neighbour_embedding))
            concept_neighbour_distance[concept] = concept_neighbour_distance.get(concept,[]) + [distance]
        index+=1

    with open(root_address + env.DATASET_NAME +"/" + endName, "w") as write_file:
        json.dump(concept_neighbour_distance, write_file)
    # return concept_neighbour_distance




# def cluster_embeddings(cluster_concepts,cluster_no,endName = "_concepts_embeddings_df.pkl"):
    
#     cluster_embeddings_list = []
#     cluster_embedding_dict = load.concept_embedding_dict()#contexual embedding edit
#     for i in cluster_concepts:
#         # cluster_embeddings_list.append(bert_embeddings.get_concept_embeddings_using_elmo(i))
#         cluster_embeddings_list.append(cluster_embedding_dict[i])#contexual embedding edit

#     cluster_concepts_embeddings_df = pd.DataFrame(cluster_embeddings_list)
#     cluster_concepts_embeddings_df.to_pickle(root_address + env.DATASET_NAME +"/" + "cluster"+str(cluster_no)+endName)



def cluster_embeddings(cluster_concepts,cluster_no,endName = "_concepts_embeddings_df.pkl"):
    
    cluster_embeddings_list = []
    cluster_embedding_dict = load.concept_embedding_dict()#contexual embedding edit
    for i in cluster_concepts:
        # cluster_embeddings_list.append(bert_embeddings.get_concept_embeddings_using_elmo(i))
        #check the type of cluster_embedding_dict[i][0]
        cluster_embeddings_list.append(cluster_embedding_dict[i][0])

    cluster_concepts_embeddings_df = pd.DataFrame(cluster_embeddings_list)
    cluster_concepts_embeddings_df.to_pickle(root_address + env.DATASET_NAME +"/" + "cluster"+str(cluster_no)+endName)


def cluster_concepts_sim(cluster_concepts_embeddings, cluster_centroid):
    cluster_concepts_simList = []
    for _,row in cluster_concepts_embeddings.iterrows():
        cluster_concepts_simList.append(1/(1+euclidean_distance(row, cluster_centroid))) #distance
        # cluster_concepts_simList.append(cosine_distance(row, cluster_centroid))  #similarity , similarity = 1 - distance
    return cluster_concepts_simList


def makeWeightDict(cluster_unique_concepts, cluster_concepts_simList):
    weight_dict = {}
    for i in range(len(cluster_unique_concepts)):
        weight_dict[cluster_unique_concepts[i]] = cluster_concepts_simList[i]
    return weight_dict

def overallDict():
    importlib.reload(load)
    unique_concepts = load.unique_concepts_for_each_cluster()
    cluster_concepts_embeddings_df = load.unique_concept_embeddings_for_each_cluster()
    clusters, centroids,_ = load.clusters_n_centroids_n_labels()

    cluster_concepts_sim_list =[]
    for  i in range(len(clusters)):
        cluster_concepts_sim_list.append(cluster_concepts_sim(cluster_concepts_embeddings_df[i], centroids[i]))

    cluster_weightDict_list = []
    for i in range(len(clusters)):
        cluster_weightDict_list.append(makeWeightDict(unique_concepts[i], cluster_concepts_sim_list[i]))

    #make a single dictionary of all clusters. Take each concept and store its weight in all clusters in a list
    cluster_weightDict = {}

    for j in range(len(clusters)):
        for i in cluster_weightDict_list[j]:
            cluster_weightDict[i] = [cluster_weightDict_list[k].get(i,0) for k in range(len(clusters))]


    with open(root_address + env.DATASET_NAME +"/" + 'cluster_weightDict.json', 'w') as fp:
        json.dump(cluster_weightDict, fp)
        



