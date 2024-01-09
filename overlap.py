#search string in a list using bisect
import bisect
import csv
import importlib
import threading
import time

pool = threading.BoundedSemaphore(30)

load = importlib.import_module("loadStuff")

import envf as env

root_address = env.root_address
overlap_summary = []


def search_list(list, string):
    i = bisect.bisect_left(list, string)
    if i != len(list) and list[i] == string:
        return i
    else:
        return -1
#find the concepts that overlap between the document and the cluster
def cluster_concepts_overlap(document_concepts, cluster_concepts):
    cluster_concepts_overlap = []
    for i in document_concepts:
        if search_list(cluster_concepts, i) != -1:
            cluster_concepts_overlap.append(i)
    return cluster_concepts_overlap


#find the concepts that are present in the neighbour concepts of each cluster
def cluster_neighbour_concepts_overlap(document_concepts, neighbourhood_dictionary):
    cluster_neighbour_concepts_overlap = []
    for i in document_concepts:
        for j in neighbourhood_dictionary.keys():
            if search_list(neighbourhood_dictionary[j], i) != -1:
                cluster_neighbour_concepts_overlap.append((i, j))
    return cluster_neighbour_concepts_overlap
# from each document extract the concepts, find overlap with cluster concepts and neighbour concepts, return count of overlap
def cluster_concepts_overlap_count(document_concept_eval, cluster_concepts, cluster_neighbour_concepts):
    cluster_concepts_overlap_direct = cluster_concepts_overlap(document_concept_eval, cluster_concepts)
    cluster_concepts_overlap_indirect = cluster_neighbour_concepts_overlap(document_concept_eval, cluster_neighbour_concepts)
    return (cluster_concepts_overlap_direct, cluster_concepts_overlap_indirect)
    


#given a document find cluster overlap with each cluster

def cluster_overlap(index,document_concept_eval,label, unique_concepts, neighbourhood_dictionaries):
    overlap_summary[index] = []
    for i in range(len(unique_concepts)):
        overlap_summary[index].append(cluster_concepts_overlap_count(document_concept_eval, unique_concepts[i], neighbourhood_dictionaries[i]))
    overlap_summary[index].append(label)
    pool.release()

#write the above function using a loop
def document_cluster_overlap(extracted_document_concepts,cluster_no,true_labels):
    global overlap_summary
    no_of_threads_already_running = threading.active_count()
    unique_concepts = load.unique_concepts_for_each_cluster()
    neighbourhood_dictionaries = load.cluster_neighbourhood_dictionaries()
    counter = 0
    overlap_summary = [None for i in range(len(extracted_document_concepts))]
    for index,i in enumerate(extracted_document_concepts):
        pool.acquire(blocking=True)
        t = threading.Thread(target=cluster_overlap, args=(index,i,true_labels[index], unique_concepts, neighbourhood_dictionaries))
        t.start()
    while threading.active_count() > no_of_threads_already_running:
        print("waiting",threading.active_count())
        time.sleep(2)
    print("done")
    
    with open(root_address + env.DATASET_NAME +"/" + "cluster_assignment_result"+str(cluster_no)+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(overlap_summary)