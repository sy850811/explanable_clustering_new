import os

import torch

infinite = 99
root_address = os.getcwd() + "/" 
crossValidationFolds = 5
baselines = ['naiveBayes','decisionTrees','randomForest']
random_states = [1,2,42,56,87]
random_state = int(os.environ['split'])
userStudyDocumentsCount = 20
userStudyTermCount = 100
# device = torch.device('cuda')
#is dataset_name_env_var == agnews

if os.environ['dataset_name_env_var'] == "agnews":
    DATASET_NAME = "agnews"
    no_of_samples = 1000
    number_of_clusters = 4
    #offline work done
elif os.environ['dataset_name_env_var'] == "dbpedia":
    DATASET_NAME = "dbpedia"
    no_of_samples = 1000
    number_of_clusters = 14
    #offline work done
elif os.environ['dataset_name_env_var'] == "r2":

    no_of_samples = 2
    number_of_clusters = 2
    DATASET_NAME = "r2"
    ##offline work done

elif os.environ['dataset_name_env_var'] == "r5":
    no_of_samples = 2
    number_of_clusters = 5
    DATASET_NAME = "r5"
    ##offline work done
elif os.environ['dataset_name_env_var'] == "yahoo":
    no_of_samples = 1000
    number_of_clusters = 10
    DATASET_NAME = "yahoo"
    ##offline work done

device = "mps" if getattr(torch,'has_mps',False) \
  else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if os.environ["cosine"] == "true":
    tmp_cosineDistance = True
else:
    tmp_cosineDistance = False

if os.environ["normalize"] == "true":
    tmp_normalize = True
else:
    tmp_normalize = False 