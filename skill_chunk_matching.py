import nltk
from nltk.corpus import stopwords
import spacy
import numpy as np
from fuzzywuzzy import fuzz
import texthero as hero



# extract noun for one job description
def extract_noun(input_text,nlp,ifchunk):
    """
    input
    input_text: string
    nlp: spaCy default nlp model
    output
    noun_list: list of nouns
    chunk_list: list of noun chunks
    """
    input_text = str(input_text).replace('\n',' ')
    input_text = " ".join([x.lower() for x in input_text.split()]) # turn all into lower case
    doc = nlp(input_text)
    noun_list = []
    chunk_list = []
    for chunk in doc.noun_chunks:
        chunk_list.append(chunk)
    for token in doc:
        if token.pos_ in ('NOUN','PROPN') and token.is_stop == False:
            noun_list.append(token.text)
    if ifchunk == 1:
        return_list =  chunk_list
    else:
        return_list =  noun_list
    return return_list


# TF-IDF Data preparation
def tfidf_input_data_format(input_pdseries,nlp,chunk):
    return input_pdseries.apply(lambda x:" ".join(extract_noun(x,nlp,chunk)))


# TF-IDF
def extracted_noun_tfidf(input_pdseries,nlp):
    """
    input
    input_pdseries: corpus
    """
    input_pdseries = tfidf_input_data_format(input_pdseries,nlp,0)
    tfidf_series = hero.tfidf(input_pdseries,return_feature_names=True)
    tfidf_list = list(tfidf_series[1])
    return tfidf_series[0].apply(lambda x:[tfidf_list[i] for i in np.argsort(x)[-0:-1]])


#chunk matching function
def chunk_matching(row_num,data_extracted_chunk,skillset): 
    l1=[str(x) for x in data_extracted_chunk[row_num]]
    matched_skill = []
    i=0
    for skill in skillset:
        for gram in l1:
            fuzzratio = fuzz.ratio(skill,gram)
            if fuzzratio >= 60:
                matched_skill.append(gram)
                break
            i+=1
            if i>10000000:
                break
    return np.unique(matched_skill)

