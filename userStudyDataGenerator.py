import importlib, json, torch
from itertools import groupby
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt


load = importlib.import_module("loadStuff")
predictionPowEval = importlib.import_module("predictionPowEval")
env = importlib.import_module("envf")
utility = importlib.import_module("utility")

root_address = env.root_address



def baseline_termCluster(N = env.userStudyTermCount):
    """
    Task 1:
    Create file named termCluster with the following format:
    Concept 1, concept 4, concept 8 .... for cluster 1 \n
    Concept 2, concept 3, concept 9 .... for cluster 2
    ...
    """
    _,_,logisticRegressionModel = load.baselineModels()
    vectorizer_step = logisticRegressionModel.named_steps['countvectorizer']
    logistic_step = logisticRegressionModel.named_steps['logisticregression']
    conceptList = vectorizer_step.get_feature_names_out()

    for k in range(env.number_of_clusters):
        conceptList = [concept.replace("_", " ") for concept in conceptList]
        
    
    with open(root_address + "userStudy/baseline_termClusters", "w") as file:
        for k in range(env.number_of_clusters):
            # Access the coefficients and sort them
            top_indices = logistic_step.coef_[k].argsort()[-N:][::-1]
            #access terms for the top indices
            sortedConceptList = [conceptList[i] for i in top_indices]
            top_indices_str = ",".join(sortedConceptList)
            file.write(top_indices_str)
            file.write("\n")
        file.close()

# def clusterNumbers():
#     with open(root_address+"userStudy/cluster.numbers","w") as file:
#         file.write(str(env.number_of_clusters))
#     file.close()

def calculate_semantic_importance(distances, vocab, cluster_index):
    return {vocab[i]: distances[i, cluster_index] for i in range(len(vocab))}

def normalize_data(data):
    max_value = max(data.values())
    return {word: score / max_value for word, score in data.items()}

def calculate_combined_scores(semantic, frequency, weight_semantic, weight_frequency):
    return {word: (weight_semantic * semantic.get(word, 0)) + (weight_frequency * frequency.get(word, 0))
            for word in set(semantic)}

def get_top_words_for_cluster(combined_scores, N):
    sorted_words = sorted(combined_scores.keys(), key=lambda w: combined_scores[w], reverse=True)
    return sorted_words[:N]

def writeTermClusterFile(top_words_per_cluster):
    """
    Task 1:
    Create file named termCluster with the following format:
    Concept 1, concept 4, concept 8 .... for cluster 1 \n
    Concept 2, concept 3, concept 9 .... for cluster 2
    ...
    """
    # Remove "_" from strings in top_words_per_cluster
    for k in range(env.number_of_clusters):
        top_words_per_cluster[k] = [word.replace("_", " ") for word in top_words_per_cluster[k]]
    with open(root_address+"userStudy/proposed_termClusters", "w") as file:
        for k in range(env.number_of_clusters):
            file.write(",".join(top_words_per_cluster[k]))
            file.write("\n")
        file.close()

def proposed_termCluster(weight_semantic = 0.8, weight_frequency = 0.2, N = env.userStudyTermCount):
    distances = np.array(load.concept_centroid_distances())  # Shape: (no. of strings) x (no. of clusters)
    vocab = load.concept_vocabulary()  # List of length (no. of strings)
    k = env.number_of_clusters
    frequency_of_concepts = utility.frequency_of_concepts_dict()  # List of k dictionaries

    # Dictionary to store top words for each cluster
    top_words_per_cluster = {}

    for cluster_index in range(k):
        semantic_importance = calculate_semantic_importance(distances, vocab, cluster_index)
        frequency = frequency_of_concepts[cluster_index]

        normalized_semantic = normalize_data(semantic_importance)
        normalized_frequency = normalize_data(frequency)

        combined_scores = calculate_combined_scores(normalized_semantic, normalized_frequency, weight_semantic, weight_frequency)

        top_representative_words = get_top_words_for_cluster(combined_scores, N)

        top_words_per_cluster[cluster_index] = top_representative_words

    writeTermClusterFile(top_words_per_cluster)

def baseline_n_proposed_documentMembers():
    """
    Task 1:
    Create 2 files named baseline_documentMembers, proposed_documentMembers with the following format: 
    name, cluster 0, cluster 1 ...
    paper0.txt, doc-clusScore, doc-clusScore ...    
    """
    
    userStudy_df = load.userStudyData()

    _,_,logisticRegressionModel = load.baselineModels()

    with open(root_address+"userStudy/baseline_documentMembers", "w") as file:
        file.write("name")
        for k in range(env.number_of_clusters):
            file.write(",cluster " + str(k))
        file.write("\n")
        for index,row in userStudy_df.iterrows():
            file.write("paper" + str(index) + ".txt")
            for k in range(env.number_of_clusters):
                file.write("," + format(logisticRegressionModel.predict_proba([row["text"]])[0][k], '.5f'))
            file.write("\n")
        file.close()

    with open(root_address+"userStudy/proposed_documentMembers", "w") as file:
        file.write("name")
        for k in range(env.number_of_clusters):
            file.write(",cluster " + str(k))
        file.write("\n")
        for index in userStudy_df.index:
            file.write("paper" + str(index) + ".txt")
            documentClusterScore = predictionPowEval.ownPredictions(userStudy_df.loc[[index]],"cosine")[0]
            #there are 4 values in documentClusterScore. We need to convert them to probabilities
            
            # documentClusterScore = softmax(torch.tensor(documentClusterScore),dim=0).tolist()
            documentClusterScore_tensor = torch.tensor(documentClusterScore).float()  # Convert to float
            documentClusterScore_softmax = softmax(documentClusterScore_tensor, dim=0).tolist()
            
            for i in documentClusterScore_softmax:
                file.write("," + format(i * 100, '.5f'))
            file.write("\n")
        file.close()

def documentClustersHelper(predictions,baselineOrProposed):
    userStudy_df = load.userStudyData()
    documentClustersString = ""
    memberCount = [len(list(group)) for key,group in groupby(sorted(predictions))]
    #i dont remember what this was for, revisit the algo and check

    for k in range(env.number_of_clusters):
        for index,_ in userStudy_df.iterrows():
            if predictions[index] == k:
                documentClustersString += "paper" + str(index) + ".txt,"
        documentClustersString = documentClustersString[:-1]
        documentClustersString += "\n"
    with open(root_address+"userStudy/"+baselineOrProposed+"_documentClusters", "w") as file:
        file.write(documentClustersString)
        file.close()

def baseline_n_proposed_documentClusters():
    """
        Task 1:
        Create file for each document named paper+index.txt
        Task 2:
        Create file named fileList with the following format:
        paper0.txt \n
        paper1.txt \n ....
        Task 3:
        filename: documentClusters
        paper0.txt,paper3.txt \n
        paper1.txt,paper2.txt,paper4.txt \n
        paper5.txt,paper6.txt,paper7.txt,paper8.txt,paper9.txt \n

    """
    userStudy_df = load.userStudyData()
    
    with open(root_address+"userStudy/fileList", "w") as fileList:
        fileName = "paper"
        for index,row in userStudy_df.iterrows():
            with open(root_address+"userStudy/"+fileName + str(index) + ".txt", "w") as file:
                file.write(row["text"])
                file.close()
            fileList.write(fileName + str(index) + ".txt\n")
        fileList.close()

    own_predictions = predictionPowEval.ownPredictions(userStudy_df,"cosine")[1]
    documentClustersHelper(own_predictions,"proposed")

    _,_,logisticRegressionModel = load.baselineModels()
    baseline_predictions = logisticRegressionModel.predict(userStudy_df["text"])
    documentClustersHelper(baseline_predictions,"baseline")

def proposed_CreateTermMembers():
    """
    Create a file of the format:
    name,cluster0,cluster1 ....
    concept1,coefofconcept1andcluster0,coefofconcept1andcluster1 ...
    concept2,coefofconcept2andcluster0,coefofconcept2andcluster1 ...
    ...
    """
    vocab = load.concept_vocabulary()
    distances = load.concept_centroid_distances()

    vocab_importance = zip(vocab,distances)
    vocab_importance = sorted(vocab_importance,key=lambda x:x[0])
    vocab,distances = zip(*vocab_importance)
    distances = np.array(distances) * 100
    print(distances.shape)
    #store in a file named proposed_termCluster
    with open("userStudy/proposed_termMembers","w") as f:
        f.write("name")
        for i in range(distances.shape[1]):
            f.write(f",cluster{i}")
        f.write("\n")
        for i in range(distances.shape[0]):
            f.write(f"{vocab[i]},")
            for j in range(distances.shape[1]):
                # only .5f
                f.write(f"{distances[i][j]:.5f},")
            f.write("\n")
def baseline_CreateTermMembers():
    """
    Create a file of the format:
    name,cluster0,cluster1 ....
    concept1,coefofconcept1andcluster0,coefofconcept1andcluster1 ...
    concept2,coefofconcept2andcluster0,coefofconcept2andcluster1 ...
    ...
    """
    _,_,logisticRegressionModel = load.baselineModels()

    logistic_step = logisticRegressionModel.steps[1][1]
    vectorizer_step = logisticRegressionModel.steps[0][1]

    vocab = vectorizer_step.get_feature_names_out()
    coef = logistic_step.coef_
    coef = coef.reshape(coef.shape[1],coef.shape[0])

    #zip vocab and coef and sort by alphabetical order of vocab
    vocab_coef = zip(vocab,coef)
    vocab_coef = sorted(vocab_coef,key=lambda x:x[0])
    vocab,coef = zip(*vocab_coef)
    coef = np.array(coef)

    #store in a file named baseline_termCluster
    with open("userStudy/baseline_termMembers","w") as f:
        f.write("name")
        for i in range(coef.shape[1]):
            f.write(f",cluster{i}")
        f.write("\n")
        for i in range(coef.shape[0]):
            concept = vocab[i].replace("_"," ")
            f.write(f"{concept},")
            for j in range(coef.shape[1]):
                # only .5f
                f.write(f"{coef[i][j]:.5f},")
            f.write("\n")

def pp_status():
    with open("userStudy/pp.status","w") as f:
        f.write("no")

def extractExplanationDetails():
    """
    Given: 
    1. Document:dataframe

    Find:
    1. Find feature importance for each concept in each document 
    2. Store the result in a json file
    """
    #load documents
    userStudy_df = load.userStudyData()
    #find explanation details with format: {document:{concept:featureImportance}}
    explaination_details = predictionPowEval.ownPredictions(userStudy_df,"cosine")[2]
    
    #store the result in a json file in userStudy folder
    with open(root_address + "userStudy/explaination_details.json", "w") as outfile:
        json.dump(explaination_details, outfile)

def visualize_local_explanation(text):
    
    """
    1. Draw visualization as decided
        - give chatgpt
            - visualisation image
            - feature importance 
            - no of clusters

    Possible improvelemnts later: 
    1. Make the feature importance as percentage
    2. Make the document assignment score as percentage as well
    3. Think of interactivity that can be introduced.(dont implement yet)
    """
    explanation_details = json.load(open(root_address + "userStudy/explaination_details.json"))
    explanation_details_data = explanation_details[text]
    # Create a figure and axis object
    _, ax = plt.subplots(figsize=(10, 6))

    # Initialize the bottom position for each stacked bar
    bottom_position = np.zeros(4)  # Explicitly define as a float array

    # Plot the bars for each concept in each cluster
    for concept, scores in explanation_details_data.items():
        ax.barh(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], scores, left=bottom_position, label=concept)
        bottom_position = bottom_position + np.array(scores)  # Ensure proper addition of float arrays

    # Adding labels and title
    ax.set_xlabel('Scores')
    ax.set_title('Scores by concept and cluster')

    # Adding legend
    ax.legend(title='Concepts', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Tight layout to fit legend
    plt.tight_layout()

    # Show the plot
    plt.show()



