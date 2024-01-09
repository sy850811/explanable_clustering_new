
import importlib
import os
from collections import Counter


def load_modules():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    return importlib.import_module("envf"),\
    importlib.import_module("loadStuff"),\
    importlib.import_module("generate_embeddings"),\
    importlib.import_module("clustering"),\
    importlib.import_module("cluster_wikifier"),\
    importlib.import_module("cluster_neighbour"),\
    importlib.import_module("concept_embedding_n_relevance"),\
    importlib.import_module("train_help_func"),\
    importlib.import_module("baselines"),\
    importlib.import_module("predictionPowEval")

def frequency_of_concepts_dict():
    env = importlib.import_module("envf")
    import numpy as np
    k = env.number_of_clusters
    # import Counter
    from collections import Counter
    load = importlib.import_module("loadStuff")
    wikifier_results = load.wikifier_concepts()
    flat_list = [[] for i in range(len(wikifier_results))]
    for i, list_of_lists in enumerate(wikifier_results):
        for sublist in list_of_lists:
            for item in sublist:
                flat_list[i].append(item)
        
    frequency_of_concepts = [{} for i in range(k)]
    for i, conceptList in enumerate(flat_list):
        list_of_tuples = list(Counter(conceptList).items())
        for tupl in list_of_tuples:
            frequency_of_concepts[i][tupl[0]] = tupl[1]

    return frequency_of_concepts

# def combine_lists (*arguments):
#     combined = []
#     for i in arguments:
#         combined.extend(i)
#     return combined

# def find_mapping(predicted_labels, true_labels):
#     from scipy.optimize import linear_sum_assignment
#     from sklearn.metrics import confusion_matrix

#     # Compute the confusion matrix between predicted and true labels
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)

#     # Convert confusion matrix to a cost matrix (negative of the confusion matrix)
#     cost_matrix = -conf_matrix

#     # Use the Hungarian algorithm to find the optimal one-to-one mapping
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)

#     # Create a dictionary to store the mapping between predicted and true labels
#     mapping = {}
#     for pred_label, true_label in zip(col_ind, row_ind):
#         mapping[pred_label] = true_label

#     return mapping

# def makeRequest():
#     import requests

#     # Step 1: Send a ping request to the server
#     ping_response = requests.get("http://www.wikifier.org")
#     if ping_response.status_code == 200:
#         print("Step 1: Server is reachable")
#     else:
#         print("Step 1: Server is not reachable")

#     # Step 2: Send a request to the server
#     response = requests.post(url="http://www.wikifier.org/annotate-article", data=[
#     ("text", "Apple"), ("lang", "en"),
#     ("userKey", "kqnkkwmvxluxwfuqjsihpirotsopzb"),
#     ("pageRankSqThreshold", "%g" % 0.85), ("applyPageRankSqThreshold", "true"),
#     ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
#     ("wikiDataClasses", "false"), ("wikiDataClassIds", "false"),
#     ("support", "false"), ("ranges", "false"), ("minLinkFrequency", "2"),
#     ("includeCosines", "false"), ("maxMentionEntropy", "3")
#     ])

#     # Step 3: Check the response
#     if response.status_code == 200:
#         print("Step 3: Request successful")
#         print(response.text)
#     else:
#         print("Step 3: Request failed with status code:", response.status_code)

# def check_trueLavelVSpredictedLabel_postClustering():
#     import loadStuff as load
#     clusters, centroids,trueLabels = load.clusters_n_centroids_n_labels()
#     wikifier_results = load.wikifier_concepts()
#     #print clusters and their corresponding true labels and predicted labels
#     true_clustering_labels = []
#     predicted_clustering_labels = []
#     for i in range(len(clusters)):
#         for j in range(len(clusters[i])):
#             true_clustering_labels.append(trueLabels[i][j])
#             predicted_clustering_labels.append(i)
#             print(clusters[i][j])
#             print("true label",trueLabels[i][j])
#             print("predicted label",i)
#             print("*********************")

#     for i in range(len(wikifier_results)):
#         for j in range(len(wikifier_results[i])):
#             print(wikifier_results[i][j])
#             print("true label",trueLabels[i][j])
#             print("predicted label",i)
#             print("*********************")
#     predicted_labels = np.array(predicted_clustering_labels)
#     true_labels = np.array(true_clustering_labels)

#     mapping = utility.find_mapping(predicted_labels, true_labels)
#     print(mapping)
# #create list of concepts from list of concept in dictionary values
# # def create_neighbour_concepts_list(cluster_neighbour_dict,no):
#     neighbour_concepts = []
#     for i in cluster_neighbour_dict.values():
#         neighbour_concepts.extend(i)
#     return list(set(neighbour_concepts))
    
# # def neighbourhoodEntityList():
#     import loadStuff as load
#     cluster_neighbour = importlib.import_module("cluster_neighbour")
#     neighbourhood_dictionary = load.cluster_neighbourhood_dictionaries()
#     cluster_unique_concepts_neighbour = []
#     for i in range(len(neighbourhood_dictionary)):
#         cluster_unique_concepts_neighbour.append(list(set(cluster_neighbour.create_neighbour_concepts_list(neighbourhood_dictionary[i],i))))

# # def getNeighbourhoodEmbedding():
#     #neighbout conecpts -> embeddings
#     pass
#     # concept_embeddings_n_relevance.concept_embedding_neighbour(cluster0_unique_concepts_neighbour,0)
#     # concept_embeddings_n_relevance.concept_embedding_neighbour(cluster1_unique_concepts_neighbour,1)
#     # concept_embeddings_n_relevance.concept_embedding_neighbour(cluster2_unique_concepts_neighbour,2)
#     # concept_embeddings_n_relevance.concept_embedding_neighbour(cluster3_unique_concepts_neighbour,3)

# # def findNoneInWikifierResults():
#     import loadStuff as load
#     wikifier_results = load.wikifier_concepts()
#     for i in range(len(wikifier_results)):
#         none = []
#         for j in range(len(wikifier_results[i])):
#             if wikifier_results[i][j] == None:
#                 none.append(j)
#         print(none)
# # def findEmptyDocumentInClusters():
#     import loadStuff as load
#     wikifier_results = load.wikifier_concepts()
#     indexes = [[] for i in range(len(wikifier_results))]
#     for i in range(len(wikifier_results)):
#         for j in range(len(wikifier_results[i])):
#             if len(wikifier_results[i][j]) == 0:
#                 indexes[i].append(j)

#keep stuff true in params to get the words used in annotation and the candidates from which we can see the prbconfidence.
# # def wikifier():
#     import requests

#     # Define the API URL
#     url = "http://www.wikifier.org/annotate-article"

#     # Define the text to be annotated
#     text = "Barrack Obama was the ex president of United States"

#     # Define the parameters for the POST request
#     params = {
#         "text": text,
#         "lang": "en",
#         "userKey": "kqnkkwmvxluxwfuqjsihpirotsopzb",
#         "pageRankSqThreshold": "0.5",
#         "applyPageRankSqThreshold": "false",
#         "nTopDfValuesToIgnore": "200",
#         "nWordsToIgnoreFromList": "200",
#         "wikiDataClasses": "false",
#         "maxTargetsPerMention": -1,
#         "wikiDataClassIds": "false",
#         "support": "true",
#         "ranges": "true",
#         "minLinkFrequency": "2",
#         "includeCosines": "true",
#         "maxMentionEntropy": "3"
#     }

#     # Send the POST request
#     response = requests.post(url, data=params)
#     for i in range(len(annotations)):
#         title = annotations[i]['title']
#         start = annotations[i]['support'][0]['chFrom']
#         end = annotations[i]['support'][0]['chTo']
#         print(title)
#         print(start,"-",end)
#         print(text.split()[start:end+1])
#         print("*********************")

#     # Check if the request was successful
#     print("hello")
#     if response.status_code == 200:
#         # Parse the JSON response
#         json_response = response.json()
#         print(json_response)

#         # Extract annotations from the JSON response
#         annotations = json_response.get("annotations", [])

#         # Extract the recognized entities and their corresponding text
#         recognized_entities = [(annotation["title"], text[annotation["chFrom"]:annotation["ranges"][0]["chTo"]])
#                             for annotation in annotations]

#         # Print the recognized entities and their corresponding text
#         for entity_title, entity_text in recognized_entities:
#             print(f"Entity Title: {entity_title}, Text: {entity_text}")
#     else:
#         print(f"Error occurred. Status Code: {response.status_code}")

# # def concept_embedding_neighbour_fast(concept_neighbour_dict, endName="cluster_neighbour_distance.json"):
#     concept_neighbour_distance = {}
#     all_strings = list(concept_neighbour_dict.keys()) + [neighbour for neighbours in concept_neighbour_dict.values() for neighbour in neighbours]
#     all_embeddings = get_concept_embeddings_using_elmo(all_strings)

#     concept_embeddings = all_embeddings[:len(concept_neighbour_dict)]
#     neighbour_embeddings = all_embeddings[len(concept_neighbour_dict):]

#     index_c = 0
#     index_n = 0
#     for concept in concept_neighbour_dict.keys():

#         concept_embedding = concept_embeddings[index_c]

#         for neighbour in concept_neighbour_dict[concept]:

#             neighbour_embedding = neighbour_embeddings[index_n]

#             distance_cn = cosine_distance(concept_embedding, neighbour_embedding)

#             concept_neighbour_distance[concept] = concept_neighbour_distance.get(concept, []) + [distance_cn]

#             index_n += 1
#         index_c += 1



#     # Store concept_neighbour_distance in a json file
#     with open(root_address + env.DATASET_NAME +"/" + endName, "w") as write_file:
#         json.dump(concept_neighbour_distance, write_file)

#     return concept_neighbour_distance
# # def compare_dict(a,b):

#     if len(a) != len(b):
#         return 1
#     for key in a.keys():
#         if key not in b.keys():
#             return 2
#         #values are list of floats
#         for i in range(len(a[key])):
#             if a[key][i] != b[key][i]:
#                 print(a[key][i],b[key][i])



# def get_data_loading_generators(chunkSize):
    # dataLoader1= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts1.csv")
    # dataLoader2= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts2.csv")
    # dataLoader3= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts3.csv")
    # dataLoader4= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts4.csv")
    # dataLoader5= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts5.csv")
    # dataLoader6= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts6.csv")
    # dataLoader7= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts7.csv")  
    # dataLoader8= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts8.csv")
    # dataLoader9= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts9.csv")
    # dataLoader10= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts10.csv")
    # dataLoader11= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts11.csv")
    # dataLoader12= generator_document_embedding_load(chunkSize,"document_embedding_wrt_concepts12.csv")

    # return dataLoader1,dataLoader2,dataLoader3,dataLoader4,dataLoader5,dataLoader6,dataLoader7,dataLoader8,dataLoader9,dataLoader10,dataLoader11,dataLoader12

#write a function to load 3000 documents at a time and store them in a file called "document_embedding_wrt_concepts1.csv" and so on
# def divide_document_embedding_wrt_concepts(batchSize = 10000):
    # counter = 1
    # with open(root_address+env.DATASET_NAME+'/'+'document_embedding_wrt_concepts.csv') as f:
    #     reader = csv.reader(f)
    #     data = []
    #     for line in reader:
    #         #each line of csv file is "[{},{},y]"
    #         # concept_dict,neighbourhood_dict,y = ast.literal_eval(line)
    #         concept_list = list(map(int, line[0].strip('[]').split(',')))
    #         neighbourhood_list = list(map(int, line[1].strip('[]').split(',')))
    #         data.append([concept_list,neighbourhood_list])
    #         if len(data) == batchSize:
    #             with open(root_address + env.DATASET_NAME +"/" +'document_embedding_wrt_concepts'+str(counter)+'.csv', 'w') as f1:
    #                 writer = csv.writer(f1)
    #                 writer.writerows(data)
    #             data = []
    #             counter += 1
    #             print(counter)
    #     print(len(data))


# def tfidfScoreCalculation():
    #Importing required module
    # wikifier_results = load.wikifier_concepts()
    # sentences = [utility.combine_lists(*cluster) for cluster in wikifier_results]
    # word_set = [entity for cluster in wikifier_results for documents in cluster for entity in documents]
    # #Set of vocab 
    # word_set = set(word_set)
    # #Total documents in our corpus
    # total_documents = len(sentences)
    
    # #Creating an index for each word in our vocab.
    # index_dict = {} #Dictionary to store index for each word
    # i = 0
    # for word in word_set:
    #     index_dict[word] = i
    #     i += 1

    # def count_dict(sentences):
    #     word_count = {}
    #     for word in word_set:
    #         word_count[word] = 0
    #         for sent in sentences:
    #             if word in sent:
    #                 word_count[word] += 1

    #     return word_count
 
    # word_count = count_dict(sentences)

    # #Term Frequency
    # def termfreq(document, word):
    #     N = len(document)
    #     occurance = len([token for token in document if token == word])
    #     return occurance/N

    # def inverse_doc_freq(word):
    #     try:
    #         word_occurance = word_count[word] + 1
    #     except:
    #         word_occurance = 1
    #     return np.log(total_documents/word_occurance)
    
    # def tf_idf(sentence):
    #     tf_idf_vec = np.zeros((len(word_set),))
    #     for word in sentence:
    #         # start = time.time()
    #         tf = termfreq(sentence,word)
    #         idf = inverse_doc_freq(word)
            
    #         value = tf*idf
    #         tf_idf_vec[index_dict[word]] = value 
    #         # end = time.time()
    #         # timeTaken = end - start
    #         # print("Time taken for one word is ",timeTaken)
    #     return tf_idf_vec
    
    # vectors = []
    # for sent in sentences:
    #     pass
    #     ###############################remove this break############################################
    #     break
    #     vec = tf_idf(sent)
    #     vectors.append(vec)
    
    # #store the vectors in a file
    # with open('tfidf_vectors.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(vectors)

    
    # return vectors

# def loadDocumentEmbeddingInChunks():
    # import loadStuff as load
    # env = importlib.import_module("envf")
    # train_help_func = importlib.import_module("train_help_func")
    # device = env.device
    # import threading
    # import time
    # from queue import Queue

    # import torch
    # pool = threading.BoundedSemaphore(14)
    # # q = Queue(maxsize=0)
    # dataLoaders = []
    # def worker(dataLoader):
    #   while True:
    #     try:
    #       q.put(next(dataLoader))
    #     except StopIteration:
    #       pool.release()
    #       break
    # def loadData(chunkSize):
    #     global q
    #     global dataLoaders
    #     dataLoaders =train_help_func.get_data_loading_generators(chunkSize)
    #     q = Queue(maxsize=0)
    #     for index,i in enumerate(dataLoaders):
    #         pool.acquire(blocking=True)
    #         t = threading.Thread(target=worker, args=(i,))
    #         t.start()
    #         time.sleep(2)



    #declare and initialize two tensor variables alpha and beta that will be trainable
    #declare and initialize two tensor variables alpha and beta that needs to retain gradient
    # datasetSize = env.no_of_samples
    # chunkSize = 2#00
    # distances = load.concept_centroid_distances()
    # distances = torch.tensor(distances).to(device)


    # # alpha = torch.tensor[0](1, requires_grad=True, device=device)#*0 + 0.8512
    # alpha = torch.tensor([0.8512], requires_grad=True, device=device)

    # beta = torch.tensor([0.2242], requires_grad=True, device=device)

    # alpha_n = torch.tensor([0.0723], requires_grad=True, device=device)

    # beta_n = torch.tensor([-0.2676], requires_grad=True, device=device)

    # #initialize alpha and beta with value 0.8512, 0.2242, 0.0723, 

    # losses = []
    # epochs = 5
    # loss_function = torch.nn.functional.binary_cross_entropy
    # optimizer = torch.optim.SGD([alpha,beta,alpha_n,beta_n], lr=0.003, momentum=0.9)
    # # torch inf
    # min_validation_loss = torch.tensor(float('inf')).to(device)
    # q = Queue(maxsize=0)
    # for i in range(epochs):
    #     batchNo = 1
    #     pool.acquire(blocking=True)
    #     t = threading.Thread(target=loadData, args=(chunkSize,))
    #     t.start()
    #     loss = torch.tensor(0.0).to(device)
    #     print(i)
    #     while True:
    #         # print("1")
    #         if batchNo >= datasetSize/chunkSize:
    #             pool.release()
    #             break

    #         #check is queue is empty
    #         waiting_counter = 0
    #         while q.empty():
    #             time.sleep(1)
    #             waiting_counter += 1
    #             if waiting_counter > 200:
    #                 break
    #         if waiting_counter > 200:
    #             #print number of threads in the pool
    #             print("No of threads: ", threading.active_count())
    #             print("datasetSize: ",datasetSize)
    #             print("chunkSize: ",chunkSize)
    #             print("batchNo: ",batchNo)
    #             for i in range(12):
    #                 try:
    #                     print("Dataloader: ",i," ",len(next(dataLoaders[i])))
    #                 except StopIteration:
    #                     print("Dataloader(how?): ",i," ",0)
    #         else:
    #             x = q.get()



    #         x = torch.tensor(q.get(),dtype=torch.float).to(device)
    #         # print("2")
    #         #extract last element of the batch which is the y
    #         cy = x[:,0,:][:,-1]
    #         cx = x[:,0,:][:,:-1]
    #         cy = torch.nn.functional.one_hot(cy.type(torch.int64) ,4).type(torch.float32)
    #         nx = x[:,1,:][:,:-1]
    #         # ny = torch.nn.functional.one_hot(y.type(torch.int64) ,4).type(torch.float32)
    #         #document concept score  = matrix multiplication of document embedding and entity concept score
    #         document_concept_score = torch.matmul(cx,torch.exp(-(alpha * distances)/10 + beta)) + torch.matmul(nx,torch.exp(-(alpha_n * distances)/100 + beta_n))
    #         #perform row wise softmax
    #         document_concept_score_dist = torch.nn.functional.softmax(document_concept_score, dim=1)
    #         #compute loss
    #         loss += loss_function(document_concept_score_dist,cy)
    #         batchNo += 1
    #     print("Out of the while loop")
    #     losses.append(loss.item())

    #     if loss < min_validation_loss:
    #         min_validation_loss = loss
    #         torch.save(alpha, 'alpha.pt')
    #         torch.save(beta, 'beta.pt')
    #         torch.save(alpha_n, 'alpha_n.pt')
    #         torch.save(beta_n, 'beta_n.pt')
    #         print("saved model")

    #     #compute gradient
    #     loss.backward()
    #     optimizer.step()
    
    # print(loss)
    # print("alph gradient",alpha.grad)
    # print("beta gradient",beta.grad)
    # print("alph_n gradient",alpha_n.grad)
    # print("beta_n gradient",beta_n.grad)
    # optimizer.zero_grad()

# def tagMe():
    # import requests

    # # Define the TagMe API URL
    # url = "https://tagme.d4science.org/tagme/tag"

    # # Define the text to be annotated
    # text = "Barrack Obama was the ex president of United States."

    # # Define your TagMe API access token (replace with your actual token)
    # access_token = "YOUR_TAGME_API_TOKEN"

    # # Define the parameters for the POST request
    # params = {
    #     "text": text,
    #     "gcube-token": '222df8c1-188e-458a-9459-bc49f43f069b-843339462',
    #     "include_abstract": False,
    #     "include_categories": False,
    #     "long_text": False,
    #     "epsilon": -3
    # }

    # # Send the POST request to the TagMe API
    # response = requests.post(url, params=params)

    # # Check if the request was successful
    # if response.status_code == 200:
    #     # Parse the JSON response
    #     json_response = response.json()
    #     print(json_response)
    #     # Extract annotations from the JSON response
    #     annotations = json_response.get("annotations", [])

    #     # Extract the recognized entities and their corresponding text
    #     recognized_entities = [(annotation["title"], annotation["spot"]) for annotation in annotations]

    #     # Print the recognized entities and their corresponding text
    #     for entity_title, entity_text in recognized_entities:
    #         print(f"Entity Title: {entity_title}, Text: {entity_text}")
    # else:
    #     print(f"Error occurred. Status Code: {response.status_code}")
# def cluster_assignment_result():
    # with open(root_address + env.DATASET_NAME +"/" + "cluster_assignment_result0.json", "r") as read_file:
    #     cluster_assignment_result0 = json.load(read_file)
    # with open(root_address + env.DATASET_NAME +"/" + "cluster_assignment_result1.json", "r") as read_file:
    #     cluster_assignment_result1 = json.load(read_file)
    # with open(root_address + env.DATASET_NAME +"/" + "cluster_assignment_result2.json", "r") as read_file:
    #     cluster_assignment_result2 = json.load(read_file)
    # with open(root_address + env.DATASET_NAME +"/" + "cluster_assignment_result3.json", "r") as read_file:
    #     cluster_assignment_result3 = json.load(read_file)

    # return cluster_assignment_result0, cluster_assignment_result1, cluster_assignment_result2, cluster_assignment_result3

# def getDocumentLabelData():
#     wikifier_results = load.wikifier_concepts()
#     documents = []
#     dl_predictions = []
#     for i in range(len(wikifier_results)):
#         for j in range(len(wikifier_results[i])):
#             doc = wikifier_results[i][j]
#             doc = [x.replace(" ","_") for x in doc]
#             documents.append(" ".join(doc))
#             dl_predictions.append(i)
#     return documents, dl_predictions

########################################from POC.ipynb######################################
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

# def removeFunctionAndUncommentCodeIfNeeded()
    """
    agnews
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
    # pass

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

