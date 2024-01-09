import pickle

from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import envf as env
import loadStuff as load
from sklearn.linear_model import LogisticRegressionCV


def getDocumentLabelData():
    """
    [
    Given: Load the documents in format [
        [                           Cluster 1
            [c1,c2,c3], #doc1
            [c1,c2,c3], #doc2
        ],                          
        [                           Cluster 2
            [c1,c2,c3], #doc1
            [c1,c2,c3], #doc2
        ],                          
        .....
    ]
    Find: 
    1) A dataset "Documents" in format 
        [
            [c1,c2,c3], #doc1
            [c1,c2,c3], #doc2
            ...
        ] 
        Note: Spaces between words of a concept are replaced with underscore "_" so that entire concept is considered as one word by baselines
    2) pseudo Labels(deep learning cluster assignments) for the dataset "dl_predictions" in format 
        [
            0, #doc1
            3, #doc2
            ... 
        ]
    ]
    """
    wikifier_results = load.wikifier_concepts()
    documents = []
    labels = []
    for i in range(len(wikifier_results)):
        for j in range(len(wikifier_results[i])):
            doc = wikifier_results[i][j]
            doc = [x.replace(" ","_") for x in doc]
            documents.append(" ".join(doc))
            labels.append(i)
    return documents, labels

def decisionTrees():
    documents, dl_predictions = getDocumentLabelData()
    train, test, train_target, test_target = train_test_split(documents, dl_predictions, test_size=0.2, random_state=env.random_state)
    model = make_pipeline(CountVectorizer(), tree.DecisionTreeClassifier())
    model.fit(train, train_target)
    print("Decision Tree accuracy wrt DL",accuracy_score(model.predict(test), test_target))
    predictions = model.predict(documents)
    return model, predictions,dl_predictions

def naiveBayes():
    documents, dl_predictions = getDocumentLabelData()
    train, test, train_target, test_target = train_test_split(documents, dl_predictions, test_size=0.2, random_state=env.random_state)
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(train, train_target)
    print("Naive Bayes accuracy wrt DL",accuracy_score(model.predict(test), test_target))
    predictions = model.predict(documents)
    return model, predictions, dl_predictions



def logisticRegression():
    documents, dl_predictions = getDocumentLabelData()
    train, test, train_target, test_target = train_test_split(documents, dl_predictions, test_size=0.2, random_state=env.random_state)
    model = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(train, train_target)
    print("Logistic Regression accuracy wrt DL",accuracy_score(model.predict(test), test_target))
    predictions = model.predict(documents)
    return model, predictions, dl_predictions


def logisticRegression():
    documents, dl_predictions = getDocumentLabelData()
    train, test, train_target, test_target = train_test_split(documents, dl_predictions, test_size=0.2, random_state=env.random_state)

    model = make_pipeline(CountVectorizer(), LogisticRegressionCV(max_iter=1000, Cs=[1], cv=5, solver='lbfgs'))


    model.fit(train, train_target)
    print("Non-Negative Logistic Regression accuracy wrt DL", accuracy_score(model.predict(test), test_target))
    predictions = model.predict(documents)
    return model, predictions, dl_predictions

def storeResults( pseudo_trueLables, predictions, decisionTreeModel, naiveBayesModel, logisticRegressionModel,decisionTreePredictions,naiveBayesPredictions,logisticRegressionPredictions):
    """
    The 
    """
    
    prefix = env.root_address + env.DATASET_NAME +"/"+env.DATASET_NAME + str(env.random_state)
    #store pseudo_trueLables.cpu(), predictions.cpu(), decisionTree model, naive bayes model, logistic regression model
    pickle.dump(pseudo_trueLables.cpu(), open( prefix + "DL.pkl", 'wb'))
    pickle.dump(predictions.cpu(), open(prefix + "proposed.pkl", 'wb'))
    pickle.dump(decisionTreeModel, open(prefix + "decisionTree.pkl", 'wb'))
    pickle.dump(naiveBayesModel, open(prefix + "naiveBayes.pkl", 'wb'))
    pickle.dump(logisticRegressionModel, open(prefix + "logisticRegression.pkl", 'wb'))
    pickle.dump(decisionTreePredictions, open(prefix + "decisionTreePredictions.pkl", 'wb'))
    pickle.dump(naiveBayesPredictions, open(prefix + "naiveBayesPredictions.pkl", 'wb'))
    pickle.dump(logisticRegressionPredictions, open(prefix + "logisticRegressionPredictions.pkl", 'wb'))




# def nbAnalysis(model):
    
#     feature_names = model.named_steps['countvectorizer'].get_feature_names_out()
#     feature_names = np.asarray(feature_names)
#     for i in range(len(model.named_steps['multinomialnb'].feature_log_prob_)):
#         topn_class1 = sorted(zip(model.named_steps['multinomialnb'].feature_log_prob_[i], feature_names),reverse=True)[:10]
#         print("Cluster ",i)
#         for coef, feat in topn_class1:
#             print(coef, feat)
#         print("\n")
