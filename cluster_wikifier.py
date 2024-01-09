import importlib
import json
import threading
import time

import requests

pool = threading.BoundedSemaphore(20)

load = importlib.import_module("loadStuff")
env = importlib.import_module("envf")
root_address = env.root_address
counter = 0

#define a function to convert a list of string into a single list
def get_unique_concepts(list_of_lists,no):
    flat_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flat_list.append(item)
    temp = sorted(list(set(flat_list)))
    #store the unique concepts in a list as json
    with open(root_address + env.DATASET_NAME +"/" + "cluster"+str(no)+"_unique_concepts.json", 'w') as f:
        json.dump(temp, f)


# Function to extract titles from the Wikifier response
def extract_titles(document_wikifier_response):
    titles = []
    for i in range(len(document_wikifier_response["annotations"])):
        titles.append(document_wikifier_response["annotations"][i]["title"])
    return titles

# Function for the worker thread
def worker(text, document_concepts):
    retrials = 1
    while True:
        if retrials % 10 == 0:
            print("text", text)

        try:
            r = requests.post(url="http://www.wikifier.org/annotate-article", data=[
                ("text", text), ("lang", "en"),
                ("userKey", "kqnkkwmvxluxwfuqjsihpirotsopzb"),
                ("pageRankSqThreshold", "%g" % 0.85), ("applyPageRankSqThreshold", "true"),
                ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
                ("wikiDataClasses", "false"), ("wikiDataClassIds", "false"),
                ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
                ("includeCosines", "false"), ("maxMentionEntropy", "3")
            ], timeout=5)
            if r.status_code == 200:
                global counter
                counter += 1
                # document_concepts[text] = extract_titles(json.loads(r.text))
                document_concepts[text] = json.loads(r.text)#contexual embedding edit
                print("done with document ",counter)
                break
            else:
                retrials += 1
                if retrials % 10 == 0:
                    print("error: Status Code", r.status_code)
                    print("error", r.text)
                time.sleep(2)  # Sleep for 2 seconds before retrying the request
        except requests.exceptions.RequestException as e:
            print("ConnectionError:", e)
            time.sleep(2)  # Sleep for 2 seconds before retrying the request

    pool.release()

def datasetWikifier():

    df,eval_df = load.test_train_dataset()
    #merge the two dataframes
    df = df.append(eval_df, ignore_index=True) #annotation is simple a dictionary of {documents:concepts}
    documents = df["text"].tolist()

    ## annotate documents
    no_of_threads_already_running = threading.active_count()
    document_concepts = {}

    for text in documents:
        pool.acquire(blocking=True)
        t = threading.Thread(target=worker, args=(text,document_concepts))
        t.start()
    print("Done")

    while threading.active_count() > no_of_threads_already_running:
        print("waiting",threading.active_count())
        time.sleep(2)
    print("done")


    ## save annotated documents
    with open(root_address + env.DATASET_NAME +"/" + "annotation.json", 'w') as f:
        json.dump(document_concepts, f)
 

def annotate_cluster_documents(): #here might error arize although unlikely
    clusters, _,_ = load.clusters_n_centroids_n_labels()
    with open(root_address + env.DATASET_NAME +"/" + "annotation.json", 'r') as f:
        annotatedDocuments = json.load(f)
    for cluster_no, _ in enumerate(clusters):
        document_concepts = []
        for document_no in range(len(clusters[cluster_no])):
            document_concepts.append(extract_titles(annotatedDocuments[clusters[cluster_no][document_no]]))
        with open(root_address + env.DATASET_NAME +"/" + "training_data_wikification"+str(cluster_no)+".json", 'w') as f:
            json.dump(document_concepts, f)
    print("done with annotations")


def getConceptSupport(r,text):

    #r is the response from wikifier
    concept_support = {}
    docWords = text.split()
    for annotation in r['annotations']:
        for support in annotation['support']:
            concept_title = annotation['title']
            concept_wFrom = support['wFrom']
            concept_wTo = support['wTo']
            wikiifierWords = r['words']
            if concept_title == "Linear Tape-Open":
                pass
            #check index of wikifierWords[concept_wFrom] is in docWords
            for ind,docword in enumerate(docWords):
                if wikiifierWords[concept_wFrom] in docword:
                    concept_wFrom = ind
                    break
            for ind,docword in enumerate(docWords[concept_wFrom:]):
                if wikiifierWords[concept_wTo] in docword:
                    concept_wTo = ind + concept_wFrom
                    break
            if concept_title in concept_support:
                concept_support[concept_title]['wordsUsed'].extend(list(range(concept_wFrom, concept_wTo+1)))
                #list(set(list1) | set(list2)) -> union of two lists
                concept_support[concept_title]['wordsUsed'] = list(set(concept_support[concept_title]['wordsUsed']) | set(list(range(concept_wFrom, concept_wTo+1))))
                largestSubString = concept_support[concept_title]['largestRangeTo']+1 - concept_support[concept_title]['largestRangeFrom']
                currentSubString = concept_wTo+1 - concept_wFrom
                if largestSubString < currentSubString:
                    concept_support[concept_title]['largestRangeFrom'] = concept_wFrom
                    concept_support[concept_title]['largestRangeTo'] = concept_wTo
            else:
                concept_support[concept_title] = {'wordsUsed': list(range(concept_wFrom, concept_wTo+1)), 'largestRangeFrom': concept_wFrom, 'largestRangeTo': concept_wTo}
    return concept_support
