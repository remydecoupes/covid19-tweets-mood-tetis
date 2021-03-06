import json
from elasticsearch import Elasticsearch
from elasticsearch import logger as es_logger
from collections import defaultdict, Counter
import re
import os
from pathlib import Path
from datetime import datetime, date
# Preprocess terms for TF-IDF
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from num2words import num2words
# end of preprocess
# LDA
from gensim import corpora, models
import pyLDAvis.gensim
# print in color
from termcolor import colored
# end LDA
import pandas as pd
import geopandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import wordnet
# SPARQL
import sparql
# progress bar
from tqdm import tqdm
# ploting
import matplotlib.pyplot as plt
from matplotlib_venn_wordcloud import venn3_wordcloud
# multiprocessing
# BERT
from transformers import pipeline
# LOG
import logging
from logging.handlers import RotatingFileHandler

def biotexInputBuilder(tweetsofcity):
    """
    Build and save a file formated for Biotex analysis
    :param tweetsofcity: dictionary of { tweets, created_at }
    :return: none
    """
    biotexcorpus = []
    for city in tweetsofcity:
        # Get all tweets for a city :
        listOfTweetsByCity = [tweets['tweet'] for tweets in tweetsofcity[city]]
        # convert this list in a big string of tweets by city
        document = '\n'.join(listOfTweetsByCity)
        biotexcorpus.append(document)
        biotexcorpus.append('\n')
        biotexcorpus.append("##########END##########")
        biotexcorpus.append('\n')
    textToSave = "".join(biotexcorpus)
    corpusfilename = "elastic-UK"
    biotexcopruspath = Path('elasticsearch/analyse')
    biotexCorpusPath = str(biotexcopruspath) + '/' + corpusfilename
    print("\t saving file : " + str(biotexCorpusPath))
    f = open(biotexCorpusPath, 'w')
    f.write(textToSave)
    f.close()


def preprocessTerms(document):
    """
    Pre process Terms according to
    https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
    /!\ Be carefull : it has a long execution time
    :param:
    :return:
    """

    def lowercase(t):
        return np.char.lower(t)

    def removesinglechar(t):
        words = word_tokenize(str(t))
        new_text = ""
        for w in words:
            if len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    def removestopwords(t):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(t))
        new_text = ""
        for w in words:
            if w not in stop_words:
                new_text = new_text + " " + w
        return new_text

    def removeapostrophe(t):
        return np.char.replace(t, "'", "")

    def removepunctuation(t):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(t, symbols[i], ' ')
            data = np.char.replace(t, "  ", " ")
        data = np.char.replace(t, ',', '')
        return data

    def convertnumbers(t):
        tokens = word_tokenize(str(t))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text

    doc = lowercase(document)
    doc = removesinglechar(doc)
    doc = removestopwords(doc)
    doc = removeapostrophe(doc)
    doc = removepunctuation(doc)
    doc = removesinglechar(doc)  # apostrophe create new single char
    return doc

def biotexAdaptativeBuilderAdaptative(listOfcities='all', spatialLevel='city', period='all', temporalLevel='day'):
    """
    Build a input biotex file well formated at the level wanted by concatenate cities's tweets
    :param listOfcities:
    :param spatialLevel:
    :param period:
    :param temporalLevel:
    :return:
    """
    matrixAggDay = pd.read_csv("elasticsearch/analyse/matrixAggDay.csv")
    # concat date with city
    matrixAggDay['city'] = matrixAggDay[['city', 'day']].agg('_'.join, axis=1)
    del matrixAggDay['day']
    ## change index
    matrixAggDay.set_index('city', inplace=True)
    matrixFiltred = spatiotemporelFilter(matrix=matrixAggDay, listOfcities=listOfcities,
                                         spatialLevel='state', period=period)

    ## Pre-process :Create 4 new columns : city, State, Country and date
    def splitindex(row):
        return row.split("_")

    matrixFiltred["city"], matrixFiltred["state"], matrixFiltred["country"], matrixFiltred["date"] = \
        zip(*matrixFiltred.index.map(splitindex))

    # Agregate by level
    if spatialLevel == 'city':
        # do nothing
        pass
    elif spatialLevel == 'state':
        matrixFiltred = matrixFiltred.groupby('state')['tweetsList'].apply('.\n'.join).reset_index()
    elif spatialLevel == 'country':
        matrixFiltred = matrixFiltred.groupby('country')['tweetsList'].apply('.\n'.join).reset_index()

    # Format biotex input file
    biotexcorpus = []
    for index, row in matrixFiltred.iterrows():
        document = row['tweetsList']
        biotexcorpus.append(document)
        biotexcorpus.append('\n')
        biotexcorpus.append("##########END##########")
        biotexcorpus.append('\n')
    textToSave = "".join(biotexcorpus)
    corpusfilename = "elastic-UK-adaptativebiotex"
    biotexcopruspath = Path('elasticsearch/analyse')
    biotexCorpusPath = str(biotexcopruspath) + '/' + corpusfilename
    print("\t saving file : " + str(biotexCorpusPath))
    f = open(biotexCorpusPath, 'w')
    f.write(textToSave)
    f.close()

def ldHHTFIDF(listOfcities):
    """ /!\ for testing only !!!!
    Only work if nb of states = nb of cities
    i.e for UK working on 4 states with their capitals...
    """
    print(colored("------------------------------------------------------------------------------------------", 'red'))
    print(colored("                                 - UNDER DEV !!! - ", 'red'))
    print(colored("------------------------------------------------------------------------------------------", 'red'))
    tfidfwords = pd.read_csv("elasticsearch/analyse/TFIDFadaptativeBiggestScore.csv", index_col=0)
    texts = pd.read_csv("elasticsearch/analyse/matrixAggDay.csv", index_col=1)
    listOfStatesTopics = []
    for i, citystate in enumerate(listOfcities):
        city = str(listOfcities[i].split("_")[0])
        state = str(listOfcities[i].split("_")[1])
        # print(str(i) + ": " + str(state) + " - " + city)
        # tfidfwords = [tfidfwords.iloc[0]]
        dictionary = corpora.Dictionary([tfidfwords.loc[state]])
        textfilter = texts.loc[texts.index.str.startswith(city + "_")]
        corpus = [dictionary.doc2bow(text.split()) for text in textfilter.tweetsList]

        # Find the better nb of topics :
        ##??Coherence measure C_v : Normalised PointWise Mutual Information (NPMI : co-occurence probability)
        ## i.e degree of semantic similarity between high scoring words in the topic
        ## and cosine similarity
        nbtopics = range(2, 35)
        coherenceScore = pd.Series(index=nbtopics, dtype=float)
        for n in nbtopics:
            lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n)
            # Compute coherence score
            ## Split each row values
            textssplit = textfilter.tweetsList.apply(lambda x: x.split()).values
            coherence = models.CoherenceModel(model=lda, texts=textssplit, dictionary=dictionary, coherence='c_v')
            coherence_result = coherence.get_coherence()
            coherenceScore[n] = coherence_result
            # print("level: " + str(state) + " - NB: " + str(n) + " - coherence LDA: " + str(coherenceScore[n]))

        # Relaunch LDA with the best nbtopic
        nbTopicOptimal = coherenceScore.idxmax()
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=nbTopicOptimal)
        # save and visualisation
        ## save
        for topic, listwords in enumerate(lda.show_topics()):
            stateTopic = {'state': state}
            ldaOuput = str(listwords).split(" + ")[1:]
            for i, word in enumerate(ldaOuput):
                # reformat lda output for each word of topics
                stateTopic[i] = ''.join(x for x in word if x.isalpha())
            listOfStatesTopics.append(stateTopic)
        ## Visualisation
        try:
            vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
            pyLDAvis.save_html(vis, "elasticsearch/analyse/lda/lda-tfidf_" + str(state) + ".html")
        except:
            print("saving pyLDAvis failed. Nb of topics for " + state + ": " + nbTopicOptimal)
    # Save file
    listOfStatesTopicsCSV = pd.DataFrame(listOfStatesTopics)
    listOfStatesTopicsCSV.to_csv("elasticsearch/analyse/lda/topicBySate.csv")

def wordnetCoverage(pdterms):
    """
    add an additionnal column with boolean term is in wordnet
    :param pdterms: pd.dataframes of terms. Must have a column with "terms" as a name
    :return: pdterms with additionnal column with boolean term is in wordnet
    """
    # Add a wordnet column boolean type : True if word is in wordnet, False otherwise
    pdterms['wordnet'] = False
    # Loop on terms and check if there are in wordnet
    for index, row in pdterms.iterrows():
        if len(wordnet.synsets(row['terms'])) != 0:
            pdterms.at[index, 'wordnet'] = True
    return pdterms


def sparqlquery(thesaurus, term):
    """
    Sparql query. This methods have be factorize to be used in case of multiprocessign
    :param thesaurus: which thesaurus to query ? agrovoc or mesh
    :param term: term to align with thesaurus
    :return: sparql result querry
    """
    # Define MeSH sparql endpoint and query
    endpointmesh = 'http://id.nlm.nih.gov/mesh/sparql'
    qmesh = (
            'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>'
            'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>'
            'PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>'
            'PREFIX owl: <http://www.w3.org/2002/07/owl#>'
            'PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>'
            'PREFIX mesh: <http://id.nlm.nih.gov/mesh/>'
            'PREFIX mesh2020: <http://id.nlm.nih.gov/mesh/2020/>'
            'PREFIX mesh2019: <http://id.nlm.nih.gov/mesh/2019/>'
            'PREFIX mesh2018: <http://id.nlm.nih.gov/mesh/2018/>'
            ''
            'ask '
            'FROM <http://id.nlm.nih.gov/mesh> '
            'WHERE { '
            '  ?meshTerms a meshv:Term .'
            '  ?meshTerms meshv:prefLabel ?label .'
            '  FILTER(lang(?label) = "en").'
            '  filter(REGEX(?label, "^' + str(term) + '$", "i"))'
                                                      ''
                                                      '}'
    )
    # Define agrovoc sparql endpoint and query
    endpointagrovoc = 'http://agrovoc.uniroma2.it/sparql'
    qagrovoc = ('PREFIX skos: <http://www.w3.org/2004/02/skos/core#> '
                'PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#> '
                'ask WHERE {'
                '?myterm skosxl:literalForm ?labelAgro.'
                'FILTER(lang(?labelAgro) = "en").'
                'filter(REGEX(?labelAgro, "^' + str(term) + '(s)*$", "i"))'
                                                            '}')
    # query mesh
    if thesaurus == "agrovoc":
        q = qagrovoc
        endpoint = endpointagrovoc
    elif thesaurus == "mesh":
        q = qmesh
        endpoint = endpointmesh
    else:
        raise Exception('Wrong thesaurus given')
    try:
        result = sparql.query(endpoint, q, timeout=30)
        # Sometimes Endpoint can bug on a request.
        # SparqlException raised by sparql-client if timeout is reach
        # other exception (That I have not identify yet) when endpoint send non well formated answer
    except:
        result = "endpoint error"
    return result


def agrovocCoverage(pdterms):
    """
    Add an additionnal column with boolean if term is in agrovoc
    :param pdterms: same as wordnetCoverage
    :return: same as wornetCoverage
    """
    # Log number of error raised by sparql endpoint
    endpointerror = 0
    # Add a agrovoc column boolean type : True if terms is in Agrovoc
    pdterms['agrovoc'] = False
    # Loop on term
    for index, row in tqdm(pdterms.iterrows(), total=pdterms.shape[0], desc="agrovoc"):
        # Build SPARQL query
        term = row['terms']
        result = sparqlquery('agrovoc', term)
        if result == "endpoint error":
            endpointerror += 1
            pdterms.at[index, 'agrovoc'] = "Error"
        elif result.hasresult():
            pdterms.at[index, 'agrovoc'] = True
    print("Agrovoc number of error: " + str(endpointerror))
    return pdterms


def meshCoverage(pdterms):
    """
    Add an additionnal column with boolean if term is in MeSH
    :param pdterms: same as wordnetCoverage
    :return: same as wornetCoverage
    """
    # Log number of error raised by sparql endpoint
    endpointerror = 0
    # Add a MeSH column boolean type : True if terms is in Mesh
    pdterms['mesh'] = False
    # Loop on term with multiprocessing
    for index, row in tqdm(pdterms.iterrows(), total=pdterms.shape[0], desc="mesh"):
        # Build SPARQL query
        term = row['terms']
        result = sparqlquery('mesh', term)
        if result == "endpoint error":
            endpointerror += 1
            pdterms.at[index, 'mesh'] = "Error"
        elif result.hasresult():
            pdterms.at[index, 'mesh'] = True
    print("Mesh number of error: " + str(endpointerror))
    return pdterms

def compareWithHTFIDF(number_of_term, dfToCompare, repToSave):
    """
    Only used for ECIR2020 not for NLDB2021

    :param number_of_term:
    :param dfToCompare:
    :param repToSave:
    :return:
    """
    # Stack / concatenate all terms from all states in one column
    HTFIDFUniquedf = concatenateHTFIDFBiggestscore()[:number_of_term]
    # select N first terms
    dfToCompare = dfToCompare[:number_of_term]
    common = pd.merge(dfToCompare, HTFIDFUniquedf, left_on='terms', right_on='terms', how='inner')
    # del common['score']
    common = common.terms.drop_duplicates()
    common = common.reset_index()
    del common['index']
    common.to_csv("elasticsearch/analyse/" + repToSave + "/common.csv")

    # Get what terms are specific to Adapt-TF-IDF
    print(dfToCompare)
    HTFIDFUniquedf['terms'][~HTFIDFUniquedf['terms'].isin(dfToCompare['terms'])].dropna()
    condition = HTFIDFUniquedf['terms'].isin(dfToCompare['terms'])
    specificHTFIDF = HTFIDFUniquedf.drop(HTFIDFUniquedf[condition].index)
    specificHTFIDF = specificHTFIDF.reset_index()
    del specificHTFIDF['index']
    specificHTFIDF.to_csv("elasticsearch/analyse/" + repToSave + "/specific-H-TFIDF.csv")

    # Get what terms are specific to dfToCompare
    dfToCompare['terms'][~dfToCompare['terms'].isin(HTFIDFUniquedf['terms'])].dropna()
    condition = dfToCompare['terms'].isin(HTFIDFUniquedf['terms'])
    specificdfToCompare = dfToCompare.drop(dfToCompare[condition].index)
    specificdfToCompare = specificdfToCompare.reset_index()
    del specificdfToCompare['index']
    specificdfToCompare.to_csv("elasticsearch/analyse/" + repToSave + "/specific-reference.csv")

    # Print stats
    percentIncommon = len(common) / len(HTFIDFUniquedf) * 100
    percentOfSpecificHTFIDF = len(specificHTFIDF) / len(HTFIDFUniquedf) * 100
    print("Percent in common " + str(percentIncommon))
    print("Percent of specific at H-TFIDF : " + str(percentOfSpecificHTFIDF))


def HTFIDF_comparewith_TFIDF_TF():
    """
    Only used for ECIR2020 not for NLDB2021
    .. warnings:: /!\ under dev !!!. See TODO below
    .. todo::
        - Remove filter and pass it as args :
            - period
            - list of Cities
        - Pass files path in args
        - Pass number of term to extract for TF-IDF and TF
    Gives commons and specifics terms between H-TFIDF and TF & TF-IDF classics
    Creates 6 csv files : 3 for each classical measures :
        - Common.csv : list of common terms
        - specific-htfidf : terms only in H-TF-IDF
        - specific-reference : terms only in one classical measurs
    """
    tfidfStartDate = date(2020, 1, 23)
    tfidfEndDate = date(2020, 1, 30)
    tfidfPeriod = pd.date_range(tfidfStartDate, tfidfEndDate)
    listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']

    # Query Elasticsearch to get all tweets from UK
    tweets = elasticsearchQuery()
    # reorganie tweets (dict : tweets by cities) into dataframe (city and date)
    col = ['tweets', 'created_at']
    matrixAllTweets = pd.DataFrame(columns=col)
    for tweetByCity in tweets.keys():
        # pprint(tweets[tweetByCity])
        # Filter cities :
        if str(tweetByCity).split("_")[0] in listOfCity:
            matrix = pd.DataFrame(tweets[tweetByCity])
            matrixAllTweets = matrixAllTweets.append(matrix, ignore_index=True)
    # NB :  28354 results instead of 44841 (from ES) because we work only on tweets with a city found
    # Split datetime into date and time
    matrixAllTweets["date"] = [d.date() for d in matrixAllTweets['created_at']]
    matrixAllTweets["time"] = [d.time() for d in matrixAllTweets['created_at']]

    # Filter by a period
    mask = ((matrixAllTweets["date"] >= tfidfPeriod.min()) & (matrixAllTweets["date"] <= tfidfPeriod.max()))
    matrixAllTweets = matrixAllTweets.loc[mask]

    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(matrixAllTweets['tweet'])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()

    ## matrixTFIDF
    TFIDFClassical = pd.DataFrame(denselist, columns=feature_names)
    ### Remove stopword
    for term in TFIDFClassical.keys():
        if term in stopwords.words('english'):
            del TFIDFClassical[term]
    # TFIDFClassical.to_csv("elasticsearch/analyse/TFIDFClassical/tfidfclassical.csv")

    ## Extract N TOP ranking score
    top_n = 500
    extractBiggest = TFIDFClassical.stack().nlargest(top_n)
    ### Reset index becaus stack create a multi-index (2 level : old index + terms)
    extractBiggest = extractBiggest.reset_index(level=[0, 1])
    extractBiggest.columns = ['old-index', 'terms', 'score']
    del extractBiggest['old-index']
    extractBiggest = extractBiggest.drop_duplicates(subset='terms', keep="first")
    extractBiggest.to_csv("elasticsearch/analyse/TFIDFClassical/TFIDFclassicalBiggestScore.csv")

    # Compare with H-TFIDF
    repToSave = "TFIDFClassical"
    compareWithHTFIDF(200, extractBiggest, repToSave)

    # Compute TF
    tf = CountVectorizer()
    tf.fit(matrixAllTweets['tweet'])
    tf_res = tf.transform(matrixAllTweets['tweet'])
    listOfTermsTF = tf.get_feature_names()
    countTerms = tf_res.todense()

    ## matrixTF
    TFClassical = pd.DataFrame(countTerms.tolist(), columns=listOfTermsTF)
    ### Remove stopword
    for term in TFClassical.keys():
        if term in stopwords.words('english'):
            del TFClassical[term]
    ### save in file
    # TFClassical.to_csv("elasticsearch/analyse/TFClassical/tfclassical.csv")

    ## Extract N TOP ranking score
    top_n = 500
    extractBiggestTF = TFClassical.stack().nlargest(top_n)
    ### Reset index becaus stack create a multi-index (2 level : old index + terms)
    extractBiggestTF = extractBiggestTF.reset_index(level=[0, 1])
    extractBiggestTF.columns = ['old-index', 'terms', 'score']
    del extractBiggestTF['old-index']
    extractBiggestTF = extractBiggestTF.drop_duplicates(subset='terms', keep="first")
    extractBiggestTF.to_csv("elasticsearch/analyse/TFClassical/TFclassicalBiggestScore.csv")

    # Compare with H-TFIDF
    repToSave = "TFClassical"
    compareWithHTFIDF(200, extractBiggestTF, repToSave)

def concatenateHTFIDFBiggestscore():
    """
    This function return a dataframe of one column containing all terms. i.e regroup all terms
    :param:
    :return: dataframe of 1 column with all terms from states stacked
    """
    HTFIDF = pd.read_csv('elasticsearch/analyse/TFIDFadaptativeBiggestScore.csv', index_col=0)
    # Transpose A-TF-IDF (inverse rows and columns)
    HTFIDF = HTFIDF.transpose()

    # group together all states' terms
    HTFIDFUnique = pd.Series(dtype='string')
    ## loop on row for append states' terms in order to take into account their rank
    ## If there are 4 states, It will add the 4 first terms by iterow
    for index, row in HTFIDF.iterrows():
        HTFIDFUnique = HTFIDFUnique.append(row.transpose(), ignore_index=True)
    ## drop duplicate
    HTFIDFUnique = HTFIDFUnique.drop_duplicates()

    # merge to see what terms have in common
    ## convert series into dataframe before merge
    HTFIDFUniquedf = HTFIDFUnique.to_frame().rename(columns={0: 'terms'})
    HTFIDFUniquedf['terms'] = HTFIDFUnique

    return HTFIDFUniquedf

def spatiotemporelFilter(matrix, listOfcities='all', spatialLevel='city', period='all', temporalLevel='day'):
    """
    Filter matrix with list of cities and a period

    :param matrix:
    :param listOfcities:
    :param spatialLevel:
    :param period:
    :param temporalLevel:
    :return: matrix filtred
    """
    if spatialLevel not in spatialLevels or temporalLevel not in temporalLevels:
        print("wrong level, please double check")
        return 1
    # Extract cities and period
    ## cities
    if listOfcities != 'all':  ### we need to filter
        ###??Initiate a numpy array of False
        filter = np.zeros((1, len(matrix.index)), dtype=bool)[0]
        for city in listOfcities:
            ### edit filter if index contains the city (for each city of the list)
            filter += matrix.index.str.startswith(str(city) + "_")
        matrix = matrix.loc[filter]
    ##??period
    if str(period) != 'all':  ### we need a filter on date
        datefilter = np.zeros((1, len(matrix.index)), dtype=bool)[0]
        for date in period:
            datefilter += matrix.index.str.contains(date.strftime('%Y-%m-%d'))
        matrix = matrix.loc[datefilter]
    return matrix



def compute_occurence_word_by_state():
    """
    Count words for tweets aggregate by state.
    For each state, we concatenate all tweets related.
    Then we build a table :
        - columns : all word (our vocabulary)
        - row : the 4 states of UK
        - cell : occurence of the word by state
    :return: pd.Dataframe of occurence of word by states
    """
    listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']
    tfidfStartDate = date(2020, 1, 23)
    tfidfEndDate = date(2020, 1, 30)
    tfidfPeriod = pd.date_range(tfidfStartDate, tfidfEndDate)

    ## Compute a table : (row : state; column: occurence of each terms present in state's tweets)
    es_tweets_results = pd.read_csv('elasticsearch/analyse/matrixOccurence.csv', index_col=0)
    es_tweets_results_filtred = spatiotemporelFilter(es_tweets_results, listOfcities=listOfCity, spatialLevel='state',
                                                     period=tfidfPeriod)

    ## Aggregate by state
    ### Create 4 new columns : city, State, Country and date
    def splitindex(row):
        return row.split("_")

    es_tweets_results_filtred["city"], es_tweets_results_filtred["state"], es_tweets_results_filtred["country"], \
    es_tweets_results_filtred["date"] = zip(*es_tweets_results_filtred.index.map(splitindex))
    es_tweets_results_filtred_aggstate = es_tweets_results_filtred.groupby("state").sum()
    return es_tweets_results_filtred_aggstate


def get_tweets_by_terms(term):
    """
    Return tweets content containing the term for Eval 11
    Warning: Only work on
        - the spatial window : capital of UK
        - the temporal windows : 2020-01-22 to 30
    Todo:
        - if you want to generelized this method at ohter spatial & temporal windows. You have to custom the
        elastic serarch query.
    :param term: term for retrieving tweets
    :return: Dictionnary of tweets for the term
    """
    list_of_tweets = []
    client = Elasticsearch("http://localhost:9200")
    index = "twitter"
    # Define a Query : Here get only city from UK
    query = {"query": {
        "bool": {
            "must": [],
            "filter": [
                {
                    "bool": {
                        "filter": [
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "bool": {
                                                "should": [
                                                    {
                                                        "match_phrase": {
                                                            "rest.features.properties.city.keyword": "London"
                                                        }
                                                    }
                                                ],
                                                "minimum_should_match": 1
                                            }
                                        },
                                        {
                                            "bool": {
                                                "should": [
                                                    {
                                                        "bool": {
                                                            "should": [
                                                                {
                                                                    "match_phrase": {
                                                                        "rest.features.properties.city.keyword": "Glasgow"
                                                                    }
                                                                }
                                                            ],
                                                            "minimum_should_match": 1
                                                        }
                                                    },
                                                    {
                                                        "bool": {
                                                            "should": [
                                                                {
                                                                    "bool": {
                                                                        "should": [
                                                                            {
                                                                                "match_phrase": {
                                                                                    "rest.features.properties.city.keyword": "Belfast"
                                                                                }
                                                                            }
                                                                        ],
                                                                        "minimum_should_match": 1
                                                                    }
                                                                },
                                                                {
                                                                    "bool": {
                                                                        "should": [
                                                                            {
                                                                                "match": {
                                                                                    "rest.features.properties.city.keyword": "Cardiff"
                                                                                }
                                                                            }
                                                                        ],
                                                                        "minimum_should_match": 1
                                                                    }
                                                                }
                                                            ],
                                                            "minimum_should_match": 1
                                                        }
                                                    }
                                                ],
                                                "minimum_should_match": 1
                                            }
                                        }
                                    ],
                                    "minimum_should_match": 1
                                }
                            },
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "match": {
                                                "full_text": term
                                            }
                                        }
                                    ],
                                    "minimum_should_match": 1
                                }
                            }
                        ]
                    }
                },
                {
                    "range": {
                        "created_at": {
                            "gte": "2020-01-22T23:00:00.000Z",
                            "lte": "2020-01-30T23:00:00.000Z",
                            "format": "strict_date_optional_time"
                        }
                    }
                }
            ],
        }
    }
    }

    try:
        result = Elasticsearch.search(client, index=index, body=query, size=10000)
    except Exception as e:
        print("Elasticsearch deamon may not be launched for term: " + term)
        print(e)
        result = ""

    for hit in result['hits']['hits']:
        content = hit["_source"]["full_text"]
        state = hit["_source"]["rest"]["features"][0]["properties"]["state"]
        tweet = {
            "full_text": content,
            "state": state
        }
        list_of_tweets.append(tweet)
    return list_of_tweets

def get_nb_of_tweets_with_spatio_temporal_filter():
    """
    Return tweets content containing the term for Eval 11
    Warning: Only work on
        - the spatial window : capital of UK
        - the temporal windows : 2020-01-22 to 30
    Todo:
        - if you want to generelized this method at ohter spatial & temporal windows. You have to custom the
        elastic serarch query.
    :param term: term for retrieving tweets
    :return: Dictionnary of nb of tweets by state
    """
    list_of_tweets = []
    client = Elasticsearch("http://localhost:9200")
    index = "twitter"
    # Define a Query : Here get only city from UK
    query = {"query": {
        "bool": {
            "must": [],
            "filter": [
                {
                    "bool": {
                        "filter": [
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "bool": {
                                                "should": [
                                                    {
                                                        "match_phrase": {
                                                            "rest.features.properties.city.keyword": "London"
                                                        }
                                                    }
                                                ],
                                                "minimum_should_match": 1
                                            }
                                        },
                                        {
                                            "bool": {
                                                "should": [
                                                    {
                                                        "bool": {
                                                            "should": [
                                                                {
                                                                    "match_phrase": {
                                                                        "rest.features.properties.city.keyword": "Glasgow"
                                                                    }
                                                                }
                                                            ],
                                                            "minimum_should_match": 1
                                                        }
                                                    },
                                                    {
                                                        "bool": {
                                                            "should": [
                                                                {
                                                                    "bool": {
                                                                        "should": [
                                                                            {
                                                                                "match_phrase": {
                                                                                    "rest.features.properties.city.keyword": "Belfast"
                                                                                }
                                                                            }
                                                                        ],
                                                                        "minimum_should_match": 1
                                                                    }
                                                                },
                                                                {
                                                                    "bool": {
                                                                        "should": [
                                                                            {
                                                                                "match": {
                                                                                    "rest.features.properties.city.keyword": "Cardiff"
                                                                                }
                                                                            }
                                                                        ],
                                                                        "minimum_should_match": 1
                                                                    }
                                                                }
                                                            ],
                                                            "minimum_should_match": 1
                                                        }
                                                    }
                                                ],
                                                "minimum_should_match": 1
                                            }
                                        }
                                    ],
                                    "minimum_should_match": 1
                                }
                            },
                        ]
                    }
                },
                {
                    "range": {
                        "created_at": {
                            "gte": "2020-01-22T23:00:00.000Z",
                            "lte": "2020-01-30T23:00:00.000Z",
                            "format": "strict_date_optional_time"
                        }
                    }
                }
            ],
        }
    }
    }

    try:
        result = Elasticsearch.search(client, index=index, body=query, size=10000)
    except Exception as e:
        print("Elasticsearch deamon may not be launched")
        print(e)
        result = ""

    nb_tweets_by_state = pd.DataFrame(index=["nb_tweets"], columns=('England', 'Northern Ireland', 'Scotland', 'Wales'))
    nb_tweets_by_state.iloc[0] = (0, 0, 0, 0)
    list_of_unboundaries_state = []
    for hit in result['hits']['hits']:
        try:
            state = hit["_source"]["rest"]["features"][0]["properties"]["state"]
            nb_tweets_by_state[state].iloc[0] += 1
        except:
            state_no_uk = str(hit["_source"]["rest"]["features"][0]["properties"]["city"] + " " + state)
            list_of_unboundaries_state.append(state_no_uk)
    print("get_nb_of_tweets_with_spatio_temporal_filter(): List of unique location outside of UK: " + str(
        set(list_of_unboundaries_state)))
    return nb_tweets_by_state

def ECIR20():
    # matrixOccurence = pd.read_csv('elasticsearch/analyse/matrixOccurence.csv', index_col=0)
    """
    ### Filter city and period
    """
    listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']
    tfidfStartDate = date(2020, 1, 23)
    tfidfEndDate = date(2020, 1, 30)
    tfidfPeriod = pd.date_range(tfidfStartDate, tfidfEndDate)
    # LDA clustering on TF-IDF adaptative vocabulary
    listOfCityState = ['London_England', 'Glasgow_Scotland', 'Belfast_Northern Ireland', 'Cardiff_Wales']
    ldHHTFIDF(listOfCityState)
    """

    """
    ## Build biotex input for adaptative level state
    biotexAdaptativeBuilderAdaptative(listOfcities=listOfCity, spatialLevel='state',
                                      period=tfidfPeriod, temporalLevel='day')
    """
    # Compare Biotex with H-TFIDF
    """
    biotex = pd.read_csv('elasticsearch/analyse/biotexonhiccs/biotexUKbyStates.csv',
                         names=['terms', 'UMLS', 'score'], sep=';')
    repToSave = "biotexonhiccs"
    compareWithHTFIDF(200, biotex, repToSave)
    """
    # declare path for comparison H-TFIDF with TF-IDF and TF (scikit measures)
    """
    tfidfpath = "elasticsearch/analyse/TFIDFClassical/TFIDFclassicalBiggestScore.csv"
    tfpath = "elasticsearch/analyse/TFClassical/TFclassicalBiggestScore.csv"
    """

    """
    # Compare classical TF-IDF with H-TFIDF
    ## HTFIDF_comparewith_TFIDF_TF() gives commun and spectific terms between H-TFIDF and TF-ISF & TF classics
    HTFIDF_comparewith_TFIDF_TF()
    """

    # Thesaurus coverage : Are the terms in Wordnet / Agrovoc / MeSH
    ## open measures results and add a column for each thesaurus
    ### TF-IDF
    """
    tfidf = pd.read_csv(tfidfpath)
    tfidf = wordnetCoverage(tfidf)
    tfidf = agrovocCoverage(tfidf)
    tfidf = meshCoverage(tfidf)
    tfidf.to_csv(tfidfpath)
    print("TF-IDF thesaurus comparison: done")

    ### TF
    tf = pd.read_csv(tfpath)
    tf = wordnetCoverage(tf)
    tf = agrovocCoverage(tf)
    tf = meshCoverage(tf)
    tf.to_csv(tfpath)
    print("TF thesaurus comparison: done")

    ### H-TFIDF
    htfidfStackedPAth = "elasticsearch/analyse/h-tfidf-stacked-wordnet.csv"
    #### Stacked H-TFIDF
    htfidf = concatenateHTFIDFBiggestscore()
    htfidf = wordnetCoverage(htfidf)
    htfidf = agrovocCoverage(htfidf)
    htfidf = meshCoverage(htfidf)
    htfidf.to_csv(htfidfStackedPAth)
    print("H-TFIDF thesaurus comparison: done")
    """

    ## Percent of Coverage : print
    """
    tfidf = pd.read_csv(tfidfpath)
    tf = pd.read_csv(tfpath)
    htfidfStackedPAth = "elasticsearch/analyse/h-tfidf-stacked-wordnet.csv"
    htfidf = pd.read_csv(htfidfStackedPAth)
    """
    """
    ### Limit to a maximun numbers of terms
    nfirstterms = 50
    ### TF-IDF
    tfidfd = tfidf[0:nfirstterms]
    tfidfPercentInWordnet = len(tfidfd[tfidfd.wordnet == True]) / nfirstterms
    print("TF-IDF wordnet coverage for the ", nfirstterms, "first terms: ", tfidfPercentInWordnet)
    tfidfPercentInAgrovoc = len(tfidfd[tfidfd.agrovoc == True]) / nfirstterms
    print("TF-IDF agrovoc coverage for the ", nfirstterms, "first terms: ", tfidfPercentInAgrovoc)
    ### TF
    tfd = tf[0:nfirstterms]
    tfPercentInWordnet = len(tfd[tfd.wordnet == True]) / nfirstterms
    print("TF wordnet coverage for the ", nfirstterms, "first terms: ", tfPercentInWordnet)
    ### H-TFIDF
    htfidfd = htfidf[0:nfirstterms]
    htfidfPercentInWordnet = len(htfidfd[htfidfd.wordnet == True]) / nfirstterms
    print("H-TFIDF wordnet coverage for the", nfirstterms, "first terms: ", htfidfPercentInWordnet)
    """

    """
    # Point 6 Comment thesaurus coverage

    ## plot graph coverage depending nb first elements
    ### Retrieve the mimimun len (i.e. nb of terms extracted) for the three measure :
    min_len = min(tfidf.shape[0], tf.shape[0], htfidf.shape[0])

    ### Building dataframes containing percent of thesaurus coverage to plot
    nbfirstelementsRange = range(1, min_len)
    col = ['h-tfidf', 'tf-idf', 'tf', 'Number_of_the_first_terms_extracted']
    wordnetCoverageByNbofterms = pd.DataFrame(columns=col)
    agrovocCoverageByBbofterms = pd.DataFrame(columns=col)
    meshCoverageByBbofterms = pd.DataFrame(columns=col)
    for i, nb in enumerate(nbfirstelementsRange):
        htfidfd = htfidf[0:nb]
        tfidfd = tfidf[0:nb]
        tfd = tf[0:nb]
        row = {
            "h-tfidf": len(htfidfd[htfidfd.wordnet == True]) / nb,
            'tf-idf': len(tfidfd[tfidfd.wordnet == True]) / nb,
            'tf': len(tfd[tfd.wordnet == True]) / nb,
            'Number_of_the_first_terms_extracted': nb
        }
        wordnetCoverageByNbofterms.loc[i] = row
        row = {
            "h-tfidf": len(htfidfd[htfidfd.agrovoc == True]) / nb,
            'tf-idf': len(tfidfd[tfidfd.agrovoc == True]) / nb,
            'tf': len(tfd[tfd.agrovoc == True]) / nb,
            'Number_of_the_first_terms_extracted': nb
        }
        agrovocCoverageByBbofterms.loc[i] = row
        row = {
            "h-tfidf": len(htfidfd[htfidfd.mesh == True]) / nb,
            'tf-idf': len(tfidfd[tfidfd.mesh == True]) / nb,
            'tf': len(tfd[tfd.mesh == True]) / nb,
            'Number_of_the_first_terms_extracted': nb
        }
        meshCoverageByBbofterms.loc[i] = row

    ### Define the figure and its axes
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].set(
        xlabel='Number of the first n elements',
        ylabel='Percentage of terms in wordnet',
        title='Wordnet'
    )
    axes[0].xaxis.set_visible(False)
    wordnetCoverageByNbofterms.plot(x='Number_of_the_first_terms_extracted', y=['h-tfidf', 'tf-idf', 'tf'], kind='line',
                                    ax=axes[0])
    axes[1].set(
        xlabel='Number of the first n elements',
        ylabel='Percentage of terms in Agrovoc',
        title='Agrovoc'
    )
    axes[1].xaxis.set_visible(False)
    agrovocCoverageByBbofterms.plot(x='Number_of_the_first_terms_extracted', y=['h-tfidf', 'tf-idf', 'tf'], kind='line',
                                    ax=axes[1])
    axes[2].set(
        xlabel='Number of the first n elements',
        ylabel='Percentage of terms in MeSH',
        title='MeSH'
    )
    # axes[2].xaxis.set_visible(False)
    meshCoverageByBbofterms.plot(x='Number_of_the_first_terms_extracted', y=['h-tfidf', 'tf-idf', 'tf'], kind='line',
                                 ax=axes[2])
    # As we hide xlabel for each subplots, we want to share one xlabel below the figure
    # fig.text(0.32, 0.04, "Number of the first n elements")
    fig.suptitle("Percentage of terms in Wordnet / Agrovoc / MesH \nby measures H-TFIDF / TF-IDF / TF")
    fig.set_size_inches(8, 15)
    # plt.show()
    # fig.savefig("elasticsearch/analyse/thesaurus_coverage.png")

    ## Venn diagram & wordcloud
    ## /!\ I have to modify source of matplotlib_venn_wordcloud/_main.py to have a good layout ...
    nb_of_terms = 99
    htfidfd = htfidf[0:nb_of_terms]
    tfidfd = tfidf[0:nb_of_terms]
    tfd = tf[0:nb_of_terms]
    ### Plot by measure, venn diagram of Wordnet / Agrovoc / MeSH
    figvenn, axvenn = plt.subplots(1, 3)
    figvenn.set_size_inches(15, 8)
    #### H-TFIDF
    sets = []
    sets.append(set(htfidfd.terms[htfidfd.wordnet == True]))
    sets.append(set(htfidfd.terms[htfidfd.agrovoc == True]))
    sets.append(set(htfidfd.terms[htfidfd.mesh == True]))
    axvenn[0].set_title("H-TFIDF Thesaurus coverage", fontsize=20)
    htfidf_ven = venn3_wordcloud(sets,
                                 set_labels=['wordnet', '  agrovoc', '             mesh'],
                                 wordcloud_kwargs=dict(min_font_size=4),
                                 ax=axvenn[0])
    for label in htfidf_ven.set_labels:
        label.set_fontsize(15)
    #### TFIDF
    sets = []
    sets.append(set(tfidfd.terms[tfidfd.wordnet == True]))
    sets.append(set(tfidfd.terms[tfidfd.agrovoc == True]))
    sets.append(set(tfidfd.terms[tfidfd.mesh == True]))
    axvenn[1].set_title("TF-IDF Thesaurus coverage", fontsize=20)
    tfidf_venn = venn3_wordcloud(sets,
                                 set_labels=['wordnet', '  agrovoc', '             mesh'],
                                 wordcloud_kwargs=dict(min_font_size=4),
                                 ax=axvenn[1])
    print(tfidf_venn.get_words_by_id("100"))
    print(tfidf_venn.get_words_by_id("110"))
    print(tfidf_venn.get_words_by_id("111"))
    print(tfidf_venn.get_words_by_id("101"))
    print(tfidfd.shape)
    for label in tfidf_venn.set_labels:
        label.set_fontsize(15)
    #### TF
    sets = []
    sets.append(set(tfd.terms[tfd.wordnet == True]))
    sets.append(set(tfd.terms[tfd.agrovoc == True]))
    sets.append(set(tfd.terms[tfd.mesh == True]))
    axvenn[2].set_title("TF Thesaurus coverage", fontsize=20)
    tf_venn = venn3_wordcloud(sets,
                              set_labels=['wordnet', '  agrovoc', '             mesh'],
                              wordcloud_kwargs=dict(min_font_size=4),
                              # wordcloud_kwargs=dict(max_font_size=10, min_font_size=10),
                              # set_colors=['r', 'g', 'b'],
                              # alpha=0.8,
                              ax=axvenn[2])
    for label in tf_venn.set_labels:
        label.set_fontsize(15)

    plt.show()

    # End of thesaurus coverage
    """

    # Point 7 : count  the number of TF / TF-IDF / H-TFIDF terms for each states
    """
    nb_of_extracted_terms_from_mesure = 300
    ## Compute a table : (row : state; column: occurence of each terms present in state's tweets)
    es_tweets_results_filtred_aggstate = compute_occurence_word_by_state()

    ## Build a table for each measures and compute nb of occurences by states
    ### TF-IDF
    tfidf_state_coverage = \
        tfidf[['terms', 'score', 'wordnet', 'agrovoc', 'mesh']].iloc[0:nb_of_extracted_terms_from_mesure]
    tfidf_state_coverage.set_index('terms', inplace=True)
    for state in es_tweets_results_filtred_aggstate.index:
        tfidf_state_coverage = \
            tfidf_state_coverage.join(es_tweets_results_filtred_aggstate.loc[state], how='left')
    tfidf_state_coverage.to_csv("elasticsearch/analyse/state_coverage/tfidf_state_coverage.csv")
    ### TF
    tf_state_coverage = \
        tf[['terms', 'score', 'wordnet', 'agrovoc', 'mesh']].iloc[0:nb_of_extracted_terms_from_mesure]
    tf_state_coverage.set_index('terms', inplace=True)
    for state in es_tweets_results_filtred_aggstate.index:
        tf_state_coverage = \
            tf_state_coverage.join(es_tweets_results_filtred_aggstate.loc[state], how='left')
    tf_state_coverage.to_csv("elasticsearch/analyse/state_coverage/tf_state_coverage.csv")
    ### H-TFIDF
    htfidf = pd.read_csv("elasticsearch/analyse/TFIDFadaptativeBiggestScore.csv", index_col=0)
    for state in es_tweets_results_filtred_aggstate.index:
        df = htfidf.loc[state].to_frame().set_index(state).join(es_tweets_results_filtred_aggstate.loc[state],
                                                                how="left")
        df.to_csv("elasticsearch/analyse/state_coverage/htfidf_" + state + ".csv")
    # end Point 7

    # Point 8 : Get K frequent terms for each state's tweets and see percentage coverage with the 3 measures
    k_first_terms = 300  # from each state get k first most frequent word
    nb_of_extracted_terms_from_mesure = 300  # from each measure, take nb first terms extract
    es_tweets_results_filtred_aggstate = compute_occurence_word_by_state()
    state_frequent_terms_by_measure_col = ["state", "terms", "occurence", "tf", "tf-idf", "h-tfidf"]
    state_frequent_terms_by_measure = \
        pd.DataFrame(columns=state_frequent_terms_by_measure_col,
                     index=range(k_first_terms * len(es_tweets_results_filtred_aggstate.index)))
    for i, state in enumerate(es_tweets_results_filtred_aggstate.index):
        state_frequent_terms_by_measure["state"][i * k_first_terms:(i + 1) * k_first_terms] = state
        state_frequent_terms_by_measure["terms"][i * k_first_terms:(i + 1) * k_first_terms] = \
            es_tweets_results_filtred_aggstate.loc[state].sort_values(ascending=False)[0:k_first_terms].index
        state_frequent_terms_by_measure["occurence"][i * k_first_terms:(i + 1) * k_first_terms] = \
            es_tweets_results_filtred_aggstate.loc[state].sort_values(ascending=False)[0:k_first_terms]
        htfidf_state = htfidf.loc[state].iloc[0:nb_of_extracted_terms_from_mesure]
        htfidf_state.rename("terms", inplace=True)
        htfidf_state = htfidf_state.to_frame().set_index("terms")
        htfidf_state["value"] = htfidf_state.index
        state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state, "h-tfidf"] = \
            state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state].join(
                htfidf_state,
                on="terms",
                how='left',
            )["value"]
        state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state, "tf"] = \
            state_frequent_terms_by_measure[state_frequent_terms_by_measure.state == state].join(
                tf.iloc[0:nb_of_extracted_terms_from_mesure].set_index("terms"),
                on="terms",
                how='left'
            )["score"]
        state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state, "tf-idf"] = \
            state_frequent_terms_by_measure[state_frequent_terms_by_measure.state == state].join(
                tfidf.iloc[0:nb_of_extracted_terms_from_mesure].set_index("terms"),
                on="terms",
                how='left'
            )["score"]
    ## save in CSV
    state_frequent_terms_by_measure.to_csv("elasticsearch/analyse/state_coverage/eval_point_8.csv")
    ## build barchart
    nb_of_k_first_terms_by_state = [100, 200, 300]
    barchart_col = ["tf", "tf-idf", "h-tfidf"]
    for nb in nb_of_k_first_terms_by_state:
        state_frequent_terms_by_measure_resize = \
            pd.DataFrame(columns=state_frequent_terms_by_measure.keys())
        for state in state_frequent_terms_by_measure.state.unique():
            state_frequent_terms_by_measure_resize = state_frequent_terms_by_measure_resize.append(
                state_frequent_terms_by_measure[state_frequent_terms_by_measure["state"] == state].iloc[0:nb],
                ignore_index=True,
            )
        barchart = pd.DataFrame(columns=barchart_col, index=range(1))
        barchart.tf = state_frequent_terms_by_measure_resize.tf.count() / len(
            state_frequent_terms_by_measure_resize) * 100
        barchart["tf-idf"] = state_frequent_terms_by_measure_resize["tf-idf"].count() / len(
            state_frequent_terms_by_measure_resize) * 100
        barchart["h-tfidf"] = state_frequent_terms_by_measure_resize["h-tfidf"].count() / len(
            state_frequent_terms_by_measure_resize) * 100
        barchart = barchart.transpose()
        # barchart.plot.bar(title="Percentage of top K first frequent terms presents in measure",
        #                   legend=False)
        barchart_by_state = state_frequent_terms_by_measure_resize.groupby(["state"]).count()
        barchart_by_state = barchart_by_state.apply(lambda x: 100 * x / nb)
        barchart_by_state[["tf", "tf-idf", "h-tfidf"]].plot.bar(
            title="Percentage of top " + str(nb) + " first frequent terms presents in measure by state",
            ylim=(0, 100)
        )
    plt.show()
    # end point 8

    # Point 9 : evaluation with TF / TF-IDF 1 doc = 1 tweet & Corpus = state

    ## Compute TF / TF-IDF by state
    # TFIDF_TF_with_corpus_state() #don't forget to launch elastic search service !!!
    ## open CSV Files
    ### H-TFIDF
    htfidf = pd.read_csv("elasticsearch/analyse/TFIDFadaptativeBiggestScore.csv", index_col=0)
    htfidf = htfidf.transpose()
    ### TF
    tf_corpus_uk = pd.read_csv("elasticsearch/analyse/TFClassical/TFclassicalBiggestScore.csv")
    tf_corpus_state = pd.read_csv("elasticsearch/analyse/point9/TF_BiggestScore.csv")
    ### TF-IDF
    tfidf_corpus_uk = pd.read_csv("elasticsearch/analyse/TFIDFClassical/TFIDFclassicalBiggestScore.csv")
    tfidf_corpus_state = pd.read_csv("elasticsearch/analyse/point9/TFIDF_BiggestScore.csv")
    ## digramm Venn between H-TFIDF and TF / TF-IDF with corpus = state
    ### Subslicing : to 400 terms
    nb_of_terms_by_states = 25
    htfidf_slice = htfidf[0:nb_of_terms_by_states]
    # tf_corpus_state_slice = tf_corpus_state[0:nb_of_terms_by_states]
    # tfidf_corpus_state_slice = tfidf_corpus_state[0:nb_of_terms_by_states]
    set_venn = pd.DataFrame(columns=["htfidf", "tfidf", "tf"], index=range(4 * nb_of_terms_by_states))
    listOfState = ["England", "Scotland", "Northern Ireland", "Wales"]
    for i, state in enumerate(listOfState):
        set_venn.loc[i * nb_of_terms_by_states:(i + 1) * nb_of_terms_by_states - 1, "htfidf"] = htfidf_slice[
            state].values
        set_venn.loc[i * nb_of_terms_by_states:(i + 1) * nb_of_terms_by_states - 1, "tfidf"] = \
            tfidf_corpus_state[tfidf_corpus_state.state == state]["terms"].values[0:nb_of_terms_by_states]
        set_venn.loc[i * nb_of_terms_by_states:(i + 1) * nb_of_terms_by_states - 1, "tf"] = \
            tf_corpus_state[tf_corpus_state.state == state]["terms"].values[0:nb_of_terms_by_states]
    ven_set = []
    for k in set_venn.keys():
        ven_set.append(set(set_venn[k].values))
    venn_corpus_state = venn3_wordcloud(ven_set,
                                        set_labels=['h-tf-idf', 'tf-idf', 'tf'],
                                        # wordcloud_kwargs=dict(min_font_size=4),
                                        # wordcloud_kwargs=dict(max_font_size=10, min_font_size=10),
                                        # set_colors=['r', 'g', 'b'],
                                        # alpha=0.8,
                                        # ax=axvenn[2]
                                        )
    for label in venn_corpus_state.set_labels:
        label.set_fontsize(15)
    # plt.show()
    ## barchart
    tf_unique = venn_corpus_state.get_words_by_id('001')
    tfidf_unique = venn_corpus_state.get_words_by_id('010')
    htfidf_unique = venn_corpus_state.get_words_by_id('100')
    barchart_col = ["tf", "tf-idf", "h-tfidf",
                    "common_tf_with_h-tfidf", "common_tf-idf_with_h-tfidf", "common_for_all",
                    "common_tf_tf-idf"]
    barchart = pd.DataFrame(columns=barchart_col, index=range(1))
    barchart.tf = len(tf_unique)
    barchart["tf-idf"] = len(tfidf_unique)
    barchart["h-tfidf"] = len(htfidf_unique)
    barchart["common_tf_with_h-tfidf"] = len(venn_corpus_state.get_words_by_id('101'))
    barchart["common_tf-idf_with_h-tfidf"] = len(venn_corpus_state.get_words_by_id('110'))
    barchart["common_for_all"] = len(venn_corpus_state.get_words_by_id('111'))
    barchart["common_tf_tf-idf"] = len(venn_corpus_state.get_words_by_id('011'))
    barchart = barchart.transpose()
    barchart.plot.bar(title="Number of specific word by measures",
                      legend=False)
    words = len(tf_unique) + len(tfidf_unique) + len(htfidf_unique) + len(venn_corpus_state.get_words_by_id('101')) + \
            len(venn_corpus_state.get_words_by_id('110')) + len(venn_corpus_state.get_words_by_id('111')) + \
            len(venn_corpus_state.get_words_by_id('011'))
    # print("nb_of_words: "+str(words))
    ## Barchart with k first occurence by step (as point 9)
    k_first_terms = 200  # from each state get k first most frequent word
    nb_of_extracted_terms_from_mesure = 100  # from each measure, take nb first terms extract
    es_tweets_results_filtred_aggstate = compute_occurence_word_by_state()
    state_frequent_terms_by_measure_col = ["state", "terms", "occurence", "tf", "tf-idf", "h-tfidf"]
    state_frequent_terms_by_measure = \
        pd.DataFrame(columns=state_frequent_terms_by_measure_col,
                     index=range(k_first_terms * len(es_tweets_results_filtred_aggstate.index)))
    for i, state in enumerate(es_tweets_results_filtred_aggstate.index):
        state_frequent_terms_by_measure["state"][i * k_first_terms:(i + 1) * k_first_terms] = state
        state_frequent_terms_by_measure["terms"][i * k_first_terms:(i + 1) * k_first_terms] = \
            es_tweets_results_filtred_aggstate.loc[state].sort_values(ascending=False)[0:k_first_terms].index
        state_frequent_terms_by_measure["occurence"][i * k_first_terms:(i + 1) * k_first_terms] = \
            es_tweets_results_filtred_aggstate.loc[state].sort_values(ascending=False)[0:k_first_terms]
        htfidf_state = htfidf[state].iloc[0:nb_of_extracted_terms_from_mesure]
        htfidf_state.rename("terms", inplace=True)
        htfidf_state = htfidf_state.to_frame().set_index("terms")
        htfidf_state["value"] = htfidf_state.index
        state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state, "h-tfidf"] = \
            state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state].join(
                htfidf_state,
                on="terms",
                how='left',
            )["value"]
        state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state, "tf"] = \
            state_frequent_terms_by_measure[state_frequent_terms_by_measure.state == state].join(
                tf_corpus_state[tf_corpus_state.state == state].iloc[0:nb_of_extracted_terms_from_mesure].set_index(
                    "terms")["score"],
                on="terms",
                how='left'
            )["score"]
        state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state, "tf-idf"] = \
            state_frequent_terms_by_measure[state_frequent_terms_by_measure.state == state].join(
                tfidf_corpus_state[tfidf_corpus_state.state == state].iloc[
                0:nb_of_extracted_terms_from_mesure].set_index("terms")["score"],
                on="terms",
                how='left'
            )["score"]
    ## save in CSV
    state_frequent_terms_by_measure.to_csv("elasticsearch/analyse/point9/state_coverage.csv")
    ## build barchart
    barchart_col = ["tf", "tf-idf", "h-tfidf"]
    barchart = pd.DataFrame(columns=barchart_col, index=range(1))
    barchart.tf = state_frequent_terms_by_measure.tf.count() / len(state_frequent_terms_by_measure) * 100
    barchart["tf-idf"] = state_frequent_terms_by_measure["tf-idf"].count() / len(state_frequent_terms_by_measure) * 100
    barchart["h-tfidf"] = state_frequent_terms_by_measure["h-tfidf"].count() / len(
        state_frequent_terms_by_measure) * 100
    barchart = barchart.transpose()
    barchart.plot.bar(title="Percentage of top K first frequent terms presents in measure",
                      legend=False)
    barchart_by_state = state_frequent_terms_by_measure.groupby(["state"]).count()
    barchart_by_state[["tf", "tf-idf", "h-tfidf"]].plot.bar(
        title="Percentage of top K first frequent terms presents in measure by state"
    )
    plt.show()
    # end point 9
    """

    """
    # Point 10
    state_coverage_corpus_uk = pd.read_csv("elasticsearch/analyse/state_coverage/eval_point_8.csv", index_col="terms")
    state_coverage_corpus_state = pd.read_csv("elasticsearch/analyse/point9/state_coverage.csv", index_col="terms")
    unique_location_uk = \
        state_coverage_corpus_uk.loc[state_coverage_corpus_uk.index.drop_duplicates(keep=False)].groupby(
            "state").count()
    unique_location_state = \
        state_coverage_corpus_state.loc[state_coverage_corpus_state.index.drop_duplicates(keep=False)].groupby(
            "state").count()
    # Normalize by number of terms (uk = 400, state = 800) and percentage
    unique_location_uk_norm = unique_location_uk * 100 / len(state_coverage_corpus_uk.index)
    unique_location_state_norm = unique_location_state * 100 / len(state_coverage_corpus_state.index)
    # Plot
    unique_location_uk_norm[["tf", "tf-idf", "h-tfidf"]].plot.bar(
        title="Percent of unique location of word retrieve by measure on corpus on whole UK for TF / TF-IDF")
    unique_location_state_norm[["tf", "tf-idf", "h-tfidf"]].plot.bar(
        title="Percent of unique location of word retrieve by measure on corpus by state for TF / TF-IDF")
    plt.show()
    # End of point 10
    """

    # Point 11
    """
    htfidf = pd.read_csv("elasticsearch/analyse/TFIDFadaptativeBiggestScore.csv", index_col=0)
    tfidf_corpus_state = pd.read_csv("elasticsearch/analyse/point9/TFIDF_BiggestScore.csv")
    htfidf = htfidf.transpose()
    ## H-TF-IDF
    for state in htfidf.keys():
        list_of_nb_tweets = []
        list_of_nb_tweets_for_concering_state = []
        list_of_nb_unique_sate = []
        list_tfidf_estimated = []
        for i, term in enumerate(htfidf[state]):
            try:
                term_tweets = get_tweets_by_terms(term)
                df = pd.DataFrame.from_dict(term_tweets)
                list_of_nb_tweets.append(len(df["full_text"]))
                list_of_nb_tweets_for_concering_state.append(len(df[df["state"] == state]))
                list_of_nb_unique_sate.append(len(df.state.unique()))
            except:
                print("error for this term: " + term)
                list_of_nb_tweets.append(np.NAN)
                list_of_nb_unique_sate.append(np.NAN)
                list_of_nb_tweets_for_concering_state.append(np.NAN)
                list_tfidf_estimated.append(np.NAN)
        htfidf[state + "_nb_tweets_with_this_term"] = list_of_nb_tweets
        htfidf[state + "_nb_tweets_for_this_state"] = list_of_nb_tweets_for_concering_state
        htfidf[state + "_nb_of_unique_state"] = list_of_nb_unique_sate
        htfidf.to_csv("elasticsearch/analyse/eval11/htfidf_nb_tweets.csv")
    # TF-IDF corpus state :
    list_of_nb_tweets = []
    list_of_nb_tweets_for_concering_state = []
    list_of_nb_unique_sate = []
    list_tfidf_estimated = []
    for term in tfidf_corpus_state.terms:
        try:
            term_tweets = get_tweets_by_terms(term)
            df = pd.DataFrame.from_dict(term_tweets)
            list_of_nb_tweets.append(len(df["full_text"]))
            list_of_nb_tweets_for_concering_state.append(
                len(df[df["state"] == tfidf_corpus_state[tfidf_corpus_state["terms"] == term].iloc[0].state]))
            list_of_nb_unique_sate.append(len(df.state.unique()))
        except:
            print("error for this term: " + term)
            list_of_nb_tweets.append(np.NAN)
            list_of_nb_unique_sate.append(np.NAN)
            list_of_nb_tweets_for_concering_state.append(np.NAN)
            list_tfidf_estimated.append(np.NAN)
    tfidf_corpus_state["nb_tweets_with_this_term"] = list_of_nb_tweets
    tfidf_corpus_state["nb_tweets_for_this_state"] = list_of_nb_tweets_for_concering_state
    tfidf_corpus_state["nb_of_unique_state"] = list_of_nb_unique_sate
    tfidf_corpus_state.to_csv("elasticsearch/analyse/eval11/tfidf_corpus_state_nb_tweets.csv")
    ## Barchart Sum of tweet for specific state / all tweet
    ###
    states = ('England', 'Northern Ireland', 'Scotland', 'Wales')
    htfidf_barchart = \
        pd.DataFrame(columns=states, index=["ratio nb of specific tweets from state / all states"])
    htfidf_barchart_dict = {}
    htfidf_mean_barchart = \
        pd.DataFrame(columns=states, index=["Mean of states"])
    htfidf_mean_barchart_dict = {}
    for state in states:
        htfidf_barchart_dict[state] = \
            htfidf[state + "_nb_tweets_for_this_state"].sum() / htfidf[state + "_nb_tweets_with_this_term"].sum() * 100
        htfidf_mean_barchart_dict[state] = htfidf[state + "_nb_of_unique_state"].mean()
    htfidf_barchart.iloc[0] = htfidf_barchart_dict
    htfidf_mean_barchart.iloc[0] = htfidf_mean_barchart_dict
    ### TF-IDF
    tfidf_grouped = tfidf_corpus_state.groupby("state")
    tfidf_barchart = \
        pd.DataFrame(columns=tfidf_grouped.groups.keys(), index=["ratio nb of specific tweets from state / all states"])
    tfidf_barchart_dict = {}
    tfidf_mean_barchart = \
        pd.DataFrame(columns=tfidf_grouped.groups.keys(), index=["Mean of states"])
    tfidf_mean_barchart_dict = {}
    for state, group in tfidf_grouped:
        tfidf_barchart_dict[state] = \
            group["nb_tweets_for_this_state"].sum() / group["nb_tweets_with_this_term"].sum() * 100
        tfidf_mean_barchart_dict[state] = group["nb_of_unique_state"].mean()
    tfidf_barchart.iloc[0] = tfidf_barchart_dict
    tfidf_mean_barchart.iloc[0] = tfidf_mean_barchart_dict

    ### Plot bar chart
    htfidf_barchart.plot.bar(
        title="H-TF-IDF Ratio of number of tweets containing terms extracted in specific state / all states")
    ax1 = plt.axes()
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)
    tfidf_barchart.plot.bar(
        title="TF-IDF Ratio of number of tweets containing terms extracted in specific state / all states")
    ax1 = plt.axes()
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)
    htfidf_mean_barchart.plot.bar(
        title="H-TF-IDF : Mean of number of states of tweets retrieved by query with H-TF-IDF extracted terms"
    )
    plt.axes().axes.get_xaxis().set_visible(False)
    plt.ylim(0, 4)
    tfidf_mean_barchart.plot.bar(
        title="TF-IDF : Mean of number of states of tweets retrieved by query with TF-IDF extracted terms"
    )
    plt.axes().axes.get_xaxis().set_visible(False)
    plt.ylim(0, 4)
    plt.show()
    """
    # End of point 11

    # Point eval12 : Choropleth Maps : https://geopandas.org/mapping.html#choropleth-maps
    """
    states = ('England', 'Northern Ireland', 'Scotland', 'Wales')
    tweets_by_state_df = get_nb_of_tweets_with_spatio_temporal_filter()
    tweets_by_state_df = tweets_by_state_df.transpose()
    print(tweets_by_state_df)
    uk_states_boundaries = geopandas.read_file("elasticsearch/analyse/eval12/uk_state_boundaries/gadm36_GBR_1.shp")
    ## Merge tweets_by_state with shapefile to get geometry
    tweets_by_state_df_geo = uk_states_boundaries.merge(tweets_by_state_df,
                                                        left_on="NAME_1",
                                                        right_index=True)[['geometry', 'nb_tweets', 'NAME_1']]
    ## Merge lost data type of column nb_tweets
    tweets_by_state_df_geo.nb_tweets = tweets_by_state_df_geo.nb_tweets.astype(int)
    tweets_by_state_df_geo.plot(column="nb_tweets", legend=True, cmap='OrRd')
    ## Display for each state on map : the number of tweet
    ### To do so, we hate to convert geometry to xy coordonnates from the figure)
    tweets_by_state_df_geo['coords'] = \
        tweets_by_state_df_geo['geometry'].apply(lambda x: x.representative_point().coords[:])
    tweets_by_state_df_geo['coords'] = [coords[0] for coords in tweets_by_state_df_geo['coords']]
    for i, row in tweets_by_state_df_geo.iterrows():
        plt.annotate(s=row["nb_tweets"], xy=row["coords"])
    plt.show()
    """
    # End of point eval12
    # Eval 13 : Bert summerization : https://huggingface.co/transformers/task_summary.html
    """
    ## Pipelin huggingface transformers :
    summarizer = pipeline("summarization")
    htfidf = pd.read_csv("elasticsearch/analyse/TFIDFadaptativeBiggestScore.csv", index_col=0)
    htfidf = htfidf.transpose()
    ## H-TF-IDF
    for state in htfidf.keys():
        list_of_tweets = []
        document = ""
        for i, term in enumerate(htfidf[state].iloc[10]):
            try:
                term_tweets = get_tweets_by_terms(term)
                df = pd.DataFrame.from_dict(term_tweets)
                list_of_tweets.append(df["full_text"].values.tolist()[0])
            except:
                print("error for this term: " + term)
                list_of_tweets.append(np.NAN)
        # print(list_of_tweets)
        document = '. '.join(list_of_tweets)
        print(summarizer(document, max_length=130, min_length=30, do_sample=False))
    # end eval 13
