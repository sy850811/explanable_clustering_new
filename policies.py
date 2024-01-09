import ast
import csv
import importlib

import numpy as np

load = importlib.import_module("loadStuff")
env = importlib.import_module("envf")
number_of_clusters = env.number_of_clusters
def load_data(filename, batch_size):
    with open (root_address + env.DATASET_NAME +"/" +filename) as f:
        reader = csv.reader(f)
        data = []
        for line in reader:
            for i in range(len(line)):
                line[i] = ast.literal_eval(line[i])
            # line = ast.literal_eval(line)
            data.append(line)
            if len(data) == batch_size:
                yield data
                data = []
        if data:
            yield data

#find assignment after scoring each cluster according to policies below
def fund_cluster_with_max_score(assignment):
    cluster_assignment_direct_weight = []
    for document in range(len(assignment)):
        try:
            cluster_assignment_direct_weight.append(assignment[document].index(max(assignment[document])))
        except:
            print(assignment[document])
    return cluster_assignment_direct_weight


#write the above function using loop
def data_loader(cluster_assignment_result_list):
    cluster_assignment_result = []
    y = []
    empty_list = 0
    for i in range(len(cluster_assignment_result_list)):
        try:
            temp = next(cluster_assignment_result_list[i])
            cluster_assignment_result.extend(temp)
            y.extend([i for i in range(len(temp))])
        except StopIteration:
            empty_list += 1

    if empty_list == len(cluster_assignment_result_list):
        #throw exception
        print("All files are empty")
        return None,None
    else:
        return cluster_assignment_result,y

# Policy - Count direct connections only
def document_cluster_score(cluster_size):

    assignment = [[None for i in range(number_of_clusters)] for j in range(cluster_size)]
    trueLabel = []
    bat_size = 1
    doc_indent = 0


    cluster_assignment_result_list = []
    for i in range(number_of_clusters):
        cluster_assignment_result_list.append(load_data('cluster_assignment_result'+str(i)+'.csv', bat_size))
    y = []
    while True:
        cluster_assignment_result,y_add = data_loader(cluster_assignment_result_list)
        if cluster_assignment_result is None:
            break
        y.extend(y_add)  
        for document in range(len(cluster_assignment_result)):
            for cluster in range(number_of_clusters):
                assignment[doc_indent][cluster] = len(cluster_assignment_result[document][cluster][0])
            doc_indent += 1
            print(doc_indent)
            trueLabel.append(cluster_assignment_result[document][-1])
    return assignment,y,trueLabel





# Policy - compute weight of the direct connection only
def document_cluster_score_weight(cluster_size,):
    cluster_weightDict = load.cluster_weightDictionary()
    assignment = [[None for i in range(number_of_clusters)] for j in range(cluster_size)]
    trueLable = []
    bat_size = 1
    doc_indent = 0
    cluster_assignment_result_list = []
    for i in range(number_of_clusters):
        cluster_assignment_result_list.append(load_data('cluster_assignment_result'+str(i)+'.csv', bat_size))

    y = []
    while True:
        cluster_assignment_result,y_add = data_loader(cluster_assignment_result_list)
        if cluster_assignment_result is None:
            break
        y.extend(y_add)  
        for document in range(len(cluster_assignment_result)):
            for cluster in range(number_of_clusters):
                weights = []
                for concept in cluster_assignment_result[document][cluster][0]:
                    if cluster_weightDict.get(concept) is not None:
                        weights.append(cluster_weightDict[concept][cluster])
                assignment[doc_indent][cluster] = sum(weights)
            trueLable.append(cluster_assignment_result[document][-1])
            doc_indent += 1
    return assignment,y,trueLable

#this policy is wrong because if elements have more weight..means it has more distance...so it should be discouraged







# Policy - compute assignment score using exponential function of weights with untrained alpha and beta
def document_cluster_exponential_score_weight(a,b,cluster_size):
    cluster_weightDict = load.cluster_weightDictionary()
    assignment = [[None for i in range(number_of_clusters)] for j in range(cluster_size)]
    trueLabel = []
    bat_size = 2
    cluster_assignment_result_list = []
    for i in range(number_of_clusters):
        cluster_assignment_result_list.append(load_data('cluster_assignment_result'+str(i)+'.csv', bat_size))

    doc_indent = 0
    y = []

    while True:
        cluster_assignment_result,y_add = data_loader(cluster_assignment_result_list)
        if cluster_assignment_result is None:
            break
        y.extend(y_add)       
        for i in cluster_assignment_result:
            for cluster in range(number_of_clusters):
                weights = np.array([])
                for concept in range(len(i[cluster][0])): # only looking at direct connection
                    if cluster_weightDict.get(i[cluster][0][concept]) is not None:
                        weights = np.append(-a * weights + b, cluster_weightDict[i[cluster][0][concept]][cluster])
                assignment[doc_indent][cluster] = sum(weights)
            doc_indent += 1
            trueLabel.append(i[-1])
        print(doc_indent)

    return assignment,y,trueLabel

