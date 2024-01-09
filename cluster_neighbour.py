import importlib
import json
import pickle
import threading
import time
import urllib

import requests

load = importlib.import_module("loadStuff")
cluster_neighbour = importlib.import_module("cluster_neighbour")
env = importlib.import_module("envf")
root_address = env.root_address
counter=0
# no_of_threads_already_running = 0 #SBcopilot
# Limit the number of threads.
pool = threading.BoundedSemaphore(50)
#write function to make url lists from concept lists

def make_url_list(concepts):
    urls = []
    for concept in concepts:
        data = urllib.parse.urlencode([("lang", 'en'), ("title", concept),
            ("nPredLevels", 0), ("nSuccLevels", 1)])
        urls.append("http://www.wikifier.org/get-neigh-graph?" + data)
    return urls

def worker(u,concept_neighbour_local_dict,v):
    while True:
        try:
            r = requests.get(u, timeout=5)
            if r.status_code == 200:
                global counter
                counter += 1
                # print("success" + str(counter))
                concept_neighbour_local_dict[v] = r.text
                pool.release()
                return
            else:
                # print("error")
                time.sleep(2)
        except Exception as e:
            print("failure :",e)




def req(concepts):
    concept_neighbour_local_dict = {}
    # Get URLs from a text file, remove white space.
    urls = make_url_list(concepts)
    no_of_threads_already_running = threading.active_count()
    for u,v in zip(urls,concepts):
        # Thread pool.
        # Blocks other threads (more than the set limit).
        pool.acquire(blocking=True)
        # Create a new thread.
        # Pass each URL (i.e. u parameter) to the worker function.
        t = threading.Thread(target=worker, args=(u,concept_neighbour_local_dict,v))
        # Start the newly create thread.
        t.start()

    #wait for all threads to finish
    while threading.active_count() > no_of_threads_already_running:
        print("waiting",threading.active_count())
        time.sleep(2)
        
    print("done fetching annotations")
    with open(root_address + env.DATASET_NAME +"/"+ "local_concept_neighbour.json", "w") as write_file:
        json.dump(concept_neighbour_local_dict, write_file)

def extract_neighbourhood_for_dataset():
    dataset_concepts = load.locally_fetch_dataset_concepts() 
    req(dataset_concepts)
    

def extract_neighbourhood_for_each_cluster():
    unique_concepts = load.unique_concepts_for_each_cluster()
    with open(root_address + env.DATASET_NAME +"/"+ "local_concept_neighbour.json", "r") as read_file:
        concept_neighbour_local_dict = json.load(read_file)
    results = [[] for i in range(len(unique_concepts))]
    for i in range(len(unique_concepts)):
        for j in range(len(unique_concepts[i])):
            try:
                results[i].append({unique_concepts[i][j]:concept_neighbour_local_dict[unique_concepts[i][j]]})
            except:
                print("error in cluster ",i)
        with open(root_address + env.DATASET_NAME +"/"+ "cluster"+str(i)+"_neighbour.json", "w") as write_file:
            json.dump(results[i], write_file)


#create a dictionary with keys stays the same and values are the titles excluding the key
def create_dict(cluster_neighbours,cluster_no):
    cluster_neighbour_dict = {}
    for c_n_dict in cluster_neighbours:
        key = list(c_n_dict.keys())[0]
        titles = json.loads(c_n_dict[key])['titles'][1:]
        cluster_neighbour_dict[key] = sorted(titles)
    with open(root_address + env.DATASET_NAME +"/"+"neighbourhood_"+str(cluster_no)+"_dictionary.json", "w") as write_file:
        json.dump(cluster_neighbour_dict, write_file)
    return cluster_neighbour_dict

def combine_dictionaries():
    neighbourhood_dictionary_list = load.cluster_neighbourhood_dictionaries()
    neighbourhood_dictionary = {}
    for i in neighbourhood_dictionary_list:
        neighbourhood_dictionary.update(i)

    with open(root_address +env.DATASET_NAME +"/"+ 'neighbourhood_dictionary.pkl', 'wb') as f:
        pickle.dump(neighbourhood_dictionary, f)
        