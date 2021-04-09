import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from rank_bm25 import BM25L
#%%
infile = open('abstract.pickle','rb')
abstracts = pickle.load(infile)
infile.close()

infile = open('title.pickle','rb')
titles = pickle.load(infile)
infile.close()

infile = open('query.pickle','rb')
queries = pickle.load(infile)
infile.close()

infile = open('question.pickle','rb')
questions = pickle.load(infile)
infile.close()

infile = open('narr.pickle','rb')
narratives = pickle.load(infile)
infile.close()

infile = open('ids.pickle','rb')
ids = pickle.load(infile)
infile.close()

infile = open('docner.pickle','rb')
doc_ners = pickle.load(infile)
infile.close()

infile = open('topicner.pickle', 'rb')
topic_ners = pickle.load(infile)
infile.close()

#%%

def jaccard(set1,set2):
    """ Calculates Jaccard coefficient between two sets."""
    if(len(set1) == 0 or len(set2) == 0):
        return 0
    return float(len(set1 & set2)) /  len(set1 | set2) 

def alljaccard(topic_ners, doc_ners):
    """ Calculates Jaccard coefficients between each document and topic."""
    table = []
    for doc in doc_ners:
        row = []
        for topic in topic_ners:
            row.append(jaccard(topic,doc))
        table.append(row)
    return np.array(table)
    
#%%
def scoreCalculate(bm25, topics):
    """ Calculates similarities between corpus and topics. """
    scores_list = []
    for topic in topics:
        doc_scores = bm25.get_scores(topic) 
        scores_list.append(doc_scores) 
    scores = np.array(scores_list)
    return np.swapaxes(scores, 0, 1) # flip the result array
    
def value_table_build():
    """ Calculates all similarity scores and builds a table from them. """
    bm_titles = BM25L(titles) # BM25 corpus from titles
    bm_abs = BM25L(abstracts) # BM25 corpus from abstracts
    
    title_query  = scoreCalculate(bm_titles, queries) # similarity scores between titles and queries
    abs_query = scoreCalculate(bm_abs, queries)
    title_question = scoreCalculate(bm_titles, questions)
    abs_question = scoreCalculate(bm_abs, questions)
    title_narr = scoreCalculate(bm_titles, narratives)
    abs_narr = scoreCalculate(bm_abs, narratives)
    jaccards = alljaccard(topic_ners,doc_ners) # jaccard similarity of scientific terms between each document and topic
    return (np.dstack([abs_query,abs_question,abs_narr,title_query,title_question,title_narr, jaccards])) # collect 7 variables in a table
    
#%%
def qrels_load():
    """ Load evaluations done by the judges."""
    URL = " https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt"
    response = requests.get(URL)
    table = []
    with open('qrels.txt', 'wb') as file:
        file.write(response.content)
    with open("qrels.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            table.append(line.split())
    qrels = pd.DataFrame(table)
    qrels.drop(1,axis=1,inplace=True) # remove unused column
    qrels.columns = ['topic','document_id','relevance'] # name the columnn
    qrels['topic'] = pd.to_numeric(qrels['topic'] ) 
    qrels['relevance'] = pd.to_numeric(qrels['relevance'] )
    return qrels

#%%

def dataset_divide(m, id_dict, train_X, train_y, train_ids, train_topics, test_X, test_y, test_ids, test_topics):
    """ Divide dataset as training and test sets """        
    if(m['topic'] % 2 == 1): # topic number is odd   
        train_X.append(id_dict[m['document_id']][m['topic']-1])
        train_y.append(m['relevance'])
        train_ids.append(m['document_id'])
        train_topics.append(m['topic'])

    
    else: # topic number is even   
        test_X.append(id_dict[m['document_id']][m['topic']-1])
        test_y.append(m['relevance'])                    
        test_ids.append(m['document_id'])
        test_topics.append(m['topic'])    



def give_output(predictions, test_ids, test_topics, test_y):
    """ Prints output file, confusion matrix and classification report. """
    outString = ""
    with open('ourPredictions.txt', 'w') as file:
        for i in range(len(test_ids)): # iterate over predictions
           
            outString +=  str(test_topics[i])+" "
            outString += "Q0 "
            outString +=  str(test_ids[i])+" 2 "
            outString +=  str(predictions[i])+" NVA\n"    
            
        file.write(outString)
        
    print(confusion_matrix(test_y,predictions))
    print(classification_report(test_y,predictions))
#%%

def main():
    train_X = [] 
    train_y = []
    test_X = []
    test_y = []
    train_ids = []
    test_ids = []
    train_topics = []
    test_topics = []

    value_table = value_table_build() # table of 7 variables, 50 topics and all selected documents
    id_dict = dict() # dictionary for locating values easily 
    for index in range(len(ids)):
        id_dict[ids[index]] = value_table[index]
    qrels = qrels_load() # load evaluations done by judges

    qrels.apply(lambda m: dataset_divide(m, id_dict, train_X, train_y, train_ids, train_topics, test_X, test_y, test_ids, test_topics) , axis = 1)
    
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    classifier = RandomForestClassifier(max_depth = 30, random_state = 300) 
    classifier.fit(train_X, train_y) # train with odd numbered topics
    predictions = classifier.predict(test_X) # predict even numbered topics 
    give_output(predictions, test_ids, test_topics, test_y) 
    
if __name__ == "__main__":
    main()