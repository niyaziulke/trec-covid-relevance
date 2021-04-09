import pandas as pd
import nltk
import string
import scispacy
import spacy
from scispacy.abbreviation import AbbreviationDetector
from nltk.stem import SnowballStemmer 
import pickle
import requests
import re
# %%
stemmer = SnowballStemmer("english") # initialize the stemmer
stopword_set = set(nltk.corpus.stopwords.words('english')) # create the set of stop words.
nlp = spacy.load("en_core_sci_sm") 
nlp.add_pipe("abbreviation_detector")  # add the abbreviation detector to nlp.
removal_string = string.punctuation+"“”" # characters to remove
#%%
def qrels_load():
    """ Loads relevance evaluations done by the judges."""
    
    URL = " https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt"
    response = requests.get(URL)
    table = []
    with open('qrels.txt', 'wb') as file:
        file.write(response.content)
    with open("qrels.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            table.append(line.split())
    return table

def docs_and_qrels():
    """ Prepares and returns the qrel relevances and the docs as dataframes"""
    qrels = pd.DataFrame(qrels_load())
    qrels.drop(1,axis=1,inplace=True)
    qrels.columns = ['topic','document_id','relevance'] # name columns
    qrels['topic'] = pd.to_numeric(qrels['topic'])  
    qrels['relevance'] = pd.to_numeric(qrels['relevance'] )
    documents = pd.read_csv("metadata.csv", low_memory=False, header=0, encoding='utf-8')
    documents =  documents[["cord_uid","title","abstract"]]
    documents.drop_duplicates(subset = ["cord_uid"],inplace = True) # drop duplicate id entries
    documents = documents[documents['cord_uid'].isin(qrels['document_id'])] # only work on documents that are evaluated by judges
    return (documents, qrels)
    
#%%
def preprocess(p_string):
    """ Lowercase and punction removal for strings in dataframe"""
    return (p_string.translate((str.maketrans(removal_string,' '*len(removal_string)))).astype(str).str.lower()) 

def documents_prepare(docs): 
    """ Preprocesses documents"""
    documents = docs.apply(lambda x: preprocess(x.astype(str).str))
    documents['title_full'] = documents['title']
    documents['title'] = documents['title'].apply(lambda x: [stemmer.stem(word) for word in x.split() if word not in stopword_set]) # stemming and stopword removal 
   
    documents['abstract'] = documents['abstract'].apply(lambda x: [stemmer.stem(word) for word in x.split() if word not in stopword_set]) # stemming and stopword removal
    return documents

#%%
def load_read_xml():
    """ Load and read information about topics"""
    URL = "https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml"
    response = requests.get(URL)
    with open('topics.xml', 'wb') as file:
        file.write(response.content)
    with open("topics.xml", "r") as file:
        topics = file.read()
    return topics

#%%

def topic_preprocess(topic_list):
    """Preprocess the topics"""
    for i in range(len(topic_list)):
        topic_list[i] = [stemmer.stem(word) for word in topic_list[i].split() if word not in stopword_set] # stopword removal and splitting
    return topic_list

def regex_finder(topics):
    """ Extracts queries, questions and narratives"""
    queries = re.findall("<query>(.*?)<\/query>", topics, flags = re.DOTALL)
    questions = re.findall("<question>(.*?)<\/question>",topics, flags=re.DOTALL)
    narratives = re.findall("<narrative>(.*?)<\/narrative>",topics, flags=re.DOTALL)
    return (queries, questions, narratives)

#%%
def pickle_all(abstracts,titles,queries,questions,narratives,ids,doc_ners,topic_ners):
    """ Pickles model parameters. """
    abstract_file  = open('abstract.pickle', 'wb') 
    title_file  = open('title.pickle', 'wb')  
    query_file  = open('query.pickle', 'wb') 
    question_file  = open('question.pickle', 'wb') 
    narr_file = open('narr.pickle', 'wb')
    id_file = open('ids.pickle','wb')
    doc_ner_file = open('docner.pickle','wb')
    topic_ner_file = open('topicner.pickle','wb')
    
    pickle.dump(abstracts, abstract_file)
    pickle.dump(titles, title_file)
    pickle.dump(queries, query_file)
    pickle.dump(questions, question_file)
    pickle.dump(narratives, narr_file)
    pickle.dump(ids, id_file)
    pickle.dump(doc_ners, doc_ner_file)
    pickle.dump(topic_ners, topic_ner_file)
#%%
abbr_dict = {}
def ner_tagger(text):
# check https://github.com/allenai/scispacy
    """ Marks named entities and stores abbreviations."""
    entities = set() # set of named entities 
    if not text:
        return entities
    doc = nlp(text)
  
    for abrv in doc._.abbreviations:
        abbr_dict[abrv] = abrv._.long_form # update abbreviations dictionary
    for ent in doc.ents:
        ner_str = str(ent).strip()
        word_list = [stemmer.stem(word)  for word in ner_str.split()] # stem named entities after detection
        ner = " ".join(word_list)
        entities.add(ner) # add named entity to set of entities        
    return entities

def abbrv_convert(set_terms):
    """ Converts abbreviations to long form. """
    for word in set_terms:
        if word in abbr_dict:
            set_terms.remove(word)
            set_terms.add(abbr_dict[word])
    return set_terms
            
#%%
def main():
    docs, qrels = docs_and_qrels() #initialize the documents and qrels
       
    ids = docs['cord_uid'].tolist()
    documents = documents_prepare(docs) #preprocess the data
        
    topics = load_read_xml() #read and store the topic file.
    
    queries, questions, narratives = regex_finder(topics) #extract queries, questions and narratives for each topic
    
    #punctuation removal and case folding for queries, questions and narratives.
    queries = [query.translate((str.maketrans(removal_string,' '*len(removal_string)))).lower() for query in queries ]
    questions = [question.translate((str.maketrans(removal_string,' '*len(removal_string)))).lower() for question in questions ]
    narratives = [narrative.translate((str.maketrans(removal_string,' '*len(removal_string)))).lower() for narrative in narratives ]
    
    topics_combined = [] 
    for i in range(len(queries)): #combine the query, question and narrative of each topic.
        topics_combined.append(queries[i] + " " + questions[i] + " " + narratives[i])
    
    topic_ners = [ner_tagger(topic) for topic in topics_combined] #do the scientific tagging operation for each topic
    
    queries = topic_preprocess(queries) #preprocess the queries
    questions = topic_preprocess(questions) #preprocess the questions
    narratives = topic_preprocess(narratives) #preprocess the narratives.
        
    abstracts = documents['abstract'].tolist() #convert the abstract column in the dataframe to a list
    titles = documents['title'].tolist() #convert the title column in the dataframe to a list
       
    documents['ners'] = documents['title_full'].apply(lambda x : ner_tagger(x)) #tag each title and store them in another column named 'ners'.
    documents['ners'] = documents['ners'].apply(lambda x : abbrv_convert(x)) #convert the abbreviations in the ners column.
    
    topic_ners = [ abbrv_convert(entity_set)  for entity_set in topic_ners] #convert the abbreviations for each topic set.
    doc_ners = documents['ners'].tolist() #convert the 'ners' column in the dataframe to a list, which will be pickled.
    pickle_all(abstracts, titles, queries, questions, narratives, ids, doc_ners, topic_ners) #pickle all the necessary data.

#%%

if __name__ == "__main__":
     main()
