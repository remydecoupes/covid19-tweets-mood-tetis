#!/usr/bin/env python

"""
analyse Elasticsearch query
"""
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
# Word embedding for evaluation
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import scipy.spatial as sp
# Spatial entity as descriptor :
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Global var on Levels on spatial and temporal axis
spatialLevels = ['city', 'state', 'country']
temporalLevels = ['day', 'week', 'month', 'period']


def elasticsearchQuery(query_fname, logger):
    """
    Build a ES query  and return a default dict with resuls
    :return: tweetsByCityAndDate
    """
    # Elastic search credentials
    client = Elasticsearch("http://localhost:9200")
    es_logger.setLevel(logging.WARNING)
    index = "twitter"
    # Define a Query
    query = open(query_fname, "r").read()
    result = Elasticsearch.search(client, index=index, body=query, scroll='2m', size=5000)

    # Append all pages form scroll search : avoid the 10k limitation of ElasticSearch
    results = avoid10kquerylimitation(result, client, logger)

    # Initiate a dict for each city append all Tweets content
    tweetsByCityAndDate = defaultdict(list)
    for hits in results:
        # parse Java date : EEE MMM dd HH:mm:ss Z yyyy
        inDate = hits["_source"]["created_at"]
        parseDate = datetime.strptime(inDate, "%a %b %d %H:%M:%S %z %Y")

        try:# geodocing may be bad
            geocoding = hits["_source"]["rest"]["features"][0]["properties"]
        except:
            continue # skip this iteraction
        if "country" in hits["_source"]["rest"]["features"][0]["properties"]:
            # locaties do not necessarily have an associated stated
            try:
                cityStateCountry = str(hits["_source"]["rest"]["features"][0]["properties"]["city"]) + "_" + \
                                   str(hits["_source"]["rest"]["features"][0]["properties"]["state"]) + "_" + \
                                   str(hits["_source"]["rest"]["features"][0]["properties"]["country"])
            except: # there is no state in geocoding
                try:
                    logger.debug(hits["_source"]["rest"]["features"][0]["properties"]["city"] + " has no state")
                    cityStateCountry = str(hits["_source"]["rest"]["features"][0]["properties"]["city"]) + "_" + \
                                       str("none") + "_" + \
                                       str(hits["_source"]["rest"]["features"][0]["properties"]["country"])
                except: # there is no city as well : only country
                    # print(json.dumps(hits["_source"], indent=4))
                    try:  #
                        cityStateCountry = str("none") + "_" + \
                                           str("none") + "_" + \
                                           str(hits["_source"]["rest"]["features"][0]["properties"]["country"])
                    except:
                        cityStateCountry = str("none") + "_" + \
                                           str("none") + "_" + \
                                           str("none")
        try:
            tweetsByCityAndDate[cityStateCountry].append(
                {
                    "tweet": preprocessTweets(hits["_source"]["full_text"]),
                    "created_at": parseDate
                }
            )
        except:
            print(json.dumps(hits["_source"], indent=4))
    # biotexInputBuilder(tweetsByCityAndDate)
    # pprint(tweetsByCityAndDate)
    return tweetsByCityAndDate


def avoid10kquerylimitation(result, client, logger):
    """
    Elasticsearch limit results of query at 10 000. To avoid this limit, we need to paginate results and scroll
    This method append all pages form scroll search
    :param result: a result of a ElasticSearcg query
    :return:
    """
    scroll_size = result['hits']['total']["value"]
    logger.info("Number of elasticsearch scroll: " + str(scroll_size))
    results = []
    # Progress bar
    pbar = tqdm(total=scroll_size)
    while (scroll_size > 0):
        try:
            scroll_id = result['_scroll_id']
            res = client.scroll(scroll_id=scroll_id, scroll='60s')
            results += res['hits']['hits']
            scroll_size = len(res['hits']['hits'])
            pbar.update(scroll_size)
        except:
            pbar.close()
            logger.error("elasticsearch search scroll failed")
            break
    pbar.close()
    return results


def preprocessTweets(text):
    """
    1 - Clean up tweets text cf : https://medium.com/analytics-vidhya/basic-tweet-preprocessing-method-with-python-56b4e53854a1
    2 - Detection lang
    3 - remove stopword ??
    :param text:
    :return: list : texclean, and langue detected
    """
    ## 1 clean up twetts
    # remove URLs
    textclean = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
    textclean = re.sub(r'http\S+', '', textclean)
    # remove usernames
    # textclean = re.sub('@[^\s]+', '', textclean)
    # remove the # in #hashtag
    # textclean = re.sub(r'#([^\s]+)', r'\1', textclean)
    return textclean


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

def matrixOccurenceBuilder(tweetsofcity, matrixAggDay_fout, matrixOccurence_fout, logger):
    """
    Create a matrix of :
        - line : (city,day)
        - column : terms
        - value of cells : TF (term frequency)
    Help found here :
    http://www.xavierdupre.fr/app/papierstat/helpsphinx/notebooks/artificiel_tokenize_features.html
    https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    :param tweetsofcity:
    :param f_out: file to save
    :return:
    """
    # initiate matrix of tweets aggregate by day
    # col = ['city', 'day', 'tweetsList', 'bow']
    col = ['city', 'day', 'tweetsList']
    matrixAggDay = pd.DataFrame(columns=col)
    cityDayList = []

    logger.info("start full_text concatenation for city & day")
    pbar = tqdm(total=len(tweetsofcity))
    for city in tweetsofcity:
        # create a table with 2 columns : tweet and created_at for a specific city
        matrix = pd.DataFrame(tweetsofcity[city])
        # Aggregate list of tweets by single day for specifics cities
        ## Loop on days for a city
        period = matrix['created_at'].dt.date
        period = period.unique()
        period.sort()
        for day in period:
            # aggregate city and date document
            document = '.\n'.join(matrix.loc[matrix['created_at'].dt.date == day]['tweet'].tolist())
            # Bag of Words and preprocces
            # preproccesFullText = preprocessTerms(document)
            tweetsOfDayAndCity = {
                'city': city,
                'day': day,
                'tweetsList': document
            }
            cityDayList.append(city + "_" + str(day))
            try:
                matrixAggDay = matrixAggDay.append(tweetsOfDayAndCity, ignore_index=True)
            except:
                print("full_text empty after pre-process: "+document)
                continue
        pbar.update(1)
    pbar.close()
    logger.info("Saving file: matrix of full_text concatenated by day & city: "+str(matrixAggDay_fout))
    matrixAggDay.to_csv(matrixAggDay_fout)

    # Count terms with sci-kit learn
    cd = CountVectorizer(
        stop_words='english',
        #preprocessor=sklearn_vectorizer_no_number_preprocessor,
        min_df=2, # token at least present in 2 cities : reduce size of matrix
        ngram_range=(1,2),
        token_pattern='[a-zA-Z0-9#]+', #remove user name, i.e term starting with @ for personnal data issue
        # strip_accents= "ascii" # remove token with special character (trying to keep only english word)
    )
    cd.fit(matrixAggDay['tweetsList'])
    res = cd.transform(matrixAggDay["tweetsList"])
    countTerms = res.todense()
    # create matrix
    ## get terms :
    # voc = cd.vocabulary_
    # listOfTerms = {term for term, index in sorted(voc.items(), key=lambda item: item[1])}
    listOfTerms = cd.get_feature_names()
    ##initiate matrix with count for each terms
    matrixOccurence = pd.DataFrame(data=countTerms[0:, 0:], index=cityDayList, columns=listOfTerms)
    ## Remove stopword
    for term in matrixOccurence.keys():
        if term in stopwords.words('english'):
            del matrixOccurence[term]

    # save to file
    logger.info("Saving file: occurence of term: "+str(matrixOccurence_fout))
    matrixOccurence.to_csv(matrixOccurence_fout)
    return matrixOccurence


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
        ### Initiate a numpy array of False
        filter = np.zeros((1, len(matrix.index)), dtype=bool)[0]
        for city in listOfcities:
            ### edit filter if index contains the city (for each city of the list)
            filter += matrix.index.str.startswith(str(city) + "_")
        matrix = matrix.loc[filter]
    ## period
    if str(period) != 'all':  ### we need a filter on date
        datefilter = np.zeros((1, len(matrix.index)), dtype=bool)[0]
        for date in period:
            datefilter += matrix.index.str.contains(date.strftime('%Y-%m-%d'))
        matrix = matrix.loc[datefilter]
    return matrix


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


def HTFIDF(matrixOcc, matrixHTFIDF_fname, biggestHTFIDFscore_fname, listOfcities='all', spatialLevel='city',
           period='all', temporalLevel='day'):
    """
    Aggregate on spatial and temporel and then compute TF-IDF

    :param matrixOcc: Matrix with TF already compute
    :param listOfcities: filter on this cities
    :param spatialLevel: city / state / country / world
    :param period: Filter on this period
    :param temporalLevel: day / week (month have to be implemented)
    :return:
    """
    matrixOcc = spatiotemporelFilter(matrix=matrixOcc, listOfcities=listOfcities,
                                     spatialLevel='state', period=period)

    # Aggregate by level
    ## Create 4 new columns : city, State, Country and date
    def splitindex(row):
        return row.split("_")

    matrixOcc["city"], matrixOcc["state"], matrixOcc["country"], matrixOcc["date"] = \
        zip(*matrixOcc.index.map(splitindex))

    if temporalLevel == 'day':
        ## In space
        if spatialLevel == 'city':
            # do nothing
            pass
        elif spatialLevel == 'state' and temporalLevel == 'day':
            matrixOcc = matrixOcc.groupby("state").sum()
        elif spatialLevel == 'country' and temporalLevel == 'day':
            matrixOcc = matrixOcc.groupby("country").sum()
    elif temporalLevel == "week":
        matrixOcc.date = pd.to_datetime((matrixOcc.date)) - pd.to_timedelta(7, unit='d')# convert date into datetime
        ## in space and time
        if spatialLevel == 'country':
            matrixOcc = matrixOcc.groupby(["country", pd.Grouper(key="date", freq="W")]).sum()
        elif spatialLevel == 'state':
            matrixOcc = matrixOcc.groupby(["state", pd.Grouper(key="date", freq="W")]).sum()
        elif spatialLevel == 'city':
            matrixOcc = matrixOcc.groupby(["city", pd.Grouper(key="date", freq="W")]).sum()


    # Compute TF-IDF
    ## compute TF : for each doc, devide count by Sum of all count
    ### Sum fo all count by row
    matrixOcc['sumCount'] = matrixOcc.sum(axis=1)
    ### Devide each cell by these sums
    listOfTerms = matrixOcc.keys()
    matrixOcc = matrixOcc.loc[:, listOfTerms].div(matrixOcc['sumCount'], axis=0)
    ## Compute IDF : create a vector of length = nb of termes with IDF value
    idf = pd.Series(index=matrixOcc.keys(), dtype=float)
    ### N : nb of doucments <=> nb of rows :
    N = matrixOcc.shape[0]
    ### DFt : nb of document that contains the term
    DFt = matrixOcc.astype(bool).sum(axis=0)  # Tip : convert all value in boolean. float O,O will be False, other True
    #### Not a Number when value 0 because otherwise log is infinite
    DFt.replace(0, np.nan, inplace=True)
    ### compute log(N/DFt)
    idf = np.log(N / DFt)
    ## compute TF-IDF
    matrixTFIDF = matrixOcc * idf
    ## remove terms if for all documents value are Nan
    matrixTFIDF.dropna(axis=1, how='all', inplace=True)

    # Save file
    matrixTFIDF.to_csv(matrixHTFIDF_fname)

    # Export N biggest TF-IDF score:
    top_n = 500
    extractBiggest = pd.DataFrame(index=matrixTFIDF.index, columns=range(0, top_n))
    for row in matrixTFIDF.index:
        try:
            row_without_zero =  matrixTFIDF.loc[row]# we remove term with a score = 0
            row_without_zero = row_without_zero[ row_without_zero !=0 ]
            try:
                extractBiggest.loc[row] = row_without_zero.nlargest(top_n).keys()
            except:
                extractBiggest.loc[row] = row_without_zero.nlargest(len(row_without_zero)).keys()
        except:
            logger.info("H-TFIDF: city "+str(matrixTFIDF.loc[row].name)+ "not enough terms")
    extractBiggest.to_csv(biggestHTFIDFscore_fname+".old.csv")
    # Transpose this table in order to share the same structure with TF-IDF classifical biggest score :
    hbt = pd.DataFrame()
    extractBiggest = extractBiggest.reset_index()
    for index, row in extractBiggest.iterrows():
        hbtrow = pd.DataFrame(row.drop([spatialLevel, "date"]).values, columns=["terms"])
        hbtrow[spatialLevel] = row[spatialLevel]
        hbtrow["date"] = row["date"]
        hbt = hbt.append(hbtrow, ignore_index=True)
    hbt.to_csv(biggestHTFIDFscore_fname)



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
        ## Coherence measure C_v : Normalised PointWise Mutual Information (NPMI : co-occurence probability)
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


def TFIDF_TF_with_corpus_state(elastic_query_fname, logger, nb_biggest_terms=500, path_for_filesaved="./",
                               spatial_hiearchy="country", temporal_period='all', listOfCities='all'):
    """
    Compute TFIDF and TF from an elastic query file
    1 doc = 1 tweet
    Corpus = by hiearchy level, i.e. : state or country

    :param elastic_query_fname: filename and path of the elastic query
    :param logger: logger of the main program
    :param nb_biggest_terms: How many biggest term are to keep
    :param spatial_hiearchy: define the size of the corpus : state or country
    :param temporal_period:
    :param listOfCities: If you want to filter out some cities, you can
    :return:
    """
    # tfidfStartDate = date(2020, 1, 23)
    # tfidfEndDate = date(2020, 1, 30)
    # temporal_period = pd.date_range(tfidfStartDate, tfidfEndDate)
    # listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']
    # listOfState = ["England", "Scotland", "Northern Ireland", "Wales"]

    # Query Elasticsearch to get all tweets from UK
    tweets = elasticsearchQuery(elastic_query_fname, logger)
    if listOfCities == 'all':
        listOfCities = []
        listOfStates = []
        listOfCountry = []
        for triple in tweetsByCityAndDate:
            splitted = triple.split("_")
            listOfCities.append(splitted[0])
            listOfStates.append(splitted[1])
            listOfCountry.append(splitted[2])
        listOfCities = list(set(listOfCities))
        listOfStates = list(set(listOfStates))
        listOfCountry = list(set(listOfCountry))
    # reorganie tweets (dict : tweets by cities) into dataframe (city and date)
    matrixAllTweets = pd.DataFrame()
    for tweetByCity in tweets.keys():
        # Filter cities :
        city = str(tweetByCity).split("_")[0]
        state = str(tweetByCity).split("_")[1]
        country = str(tweetByCity).split("_")[2]
        if city in listOfCities:
            matrix = pd.DataFrame(tweets[tweetByCity])
            matrix['city'] = city
            matrix['state'] = state
            matrix['country'] = country
            matrixAllTweets = matrixAllTweets.append(matrix, ignore_index=True)

    # Split datetime into date and time
    matrixAllTweets["date"] = [d.date() for d in matrixAllTweets['created_at']]
    matrixAllTweets["time"] = [d.time() for d in matrixAllTweets['created_at']]

    # Filter by a period
    if temporal_period != "all":
        mask = ((matrixAllTweets["date"] >= temporal_period.min()) & (matrixAllTweets["date"] <= temporal_period.max()))
        matrixAllTweets = matrixAllTweets.loc[mask]

    # Compute TF-IDF and TF by state
    extractBiggestTF_allstates = pd.DataFrame()
    extractBiggestTFIDF_allstates = pd.DataFrame()

    if spatial_hiearchy == "country":
        listOfLocalities = listOfCountry
    elif spatial_hiearchy == "state":
        listOfLocalities = listOfStates
    elif spatial_hiearchy == "city":
        listOfLocalities = listOfCities

    for locality in listOfLocalities:
        matrix_by_locality = matrixAllTweets[matrixAllTweets[spatial_hiearchy] == locality]
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=0.001,
            ngram_range=(1, 2),
            token_pattern='[a-zA-Z0-9#]+', #remove user name, i.e term starting with @ for personnal data issue
        )
        # logger.info("Compute TF-IDF on corpus = "+spatial_hiearchy)
        try:
            vectors = vectorizer.fit_transform(matrix_by_locality['tweet'])
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
        except:
            logger.info("Impossible to compute TF-IDF on: "+locality)
            continue
        ## matrixTFIDF
        TFIDFClassical = pd.DataFrame(denselist, columns=feature_names)
        locality_format = locality.replace("/", "_")
        locality_format = locality_format.replace(" ", "_")
        logger.info("saving TF-IDF File: "+path_for_filesaved+"/tfidf_on_"+locality_format+"_corpus.csv")
        TFIDFClassical.to_csv(path_for_filesaved+"/tfidf_on_"+locality_format+"_corpus.csv")
        ## Extract N TOP ranking score
        extractBiggest = TFIDFClassical.max().nlargest(nb_biggest_terms)
        extractBiggest = extractBiggest.to_frame()
        extractBiggest = extractBiggest.reset_index()
        extractBiggest.columns = ['terms', 'score']
        extractBiggest[spatial_hiearchy] = locality
        extractBiggestTFIDF_allstates = extractBiggestTFIDF_allstates.append(extractBiggest, ignore_index=True)

        """
        # Compute TF
        tf = CountVectorizer(
            stop_words='english',
            min_df=2,
            ngram_range=(1,2),
            token_pattern='[a-zA-Z0-9@#]+',
        )
        try:
            tf.fit(matrix_by_locality['tweet'])
            tf_res = tf.transform(matrix_by_locality['tweet'])
            listOfTermsTF = tf.get_feature_names()
            countTerms = tf_res.todense()
        except:# locality does not have enough different term
            logger.info("Impossible to compute TF on: "+locality)
            continue
        ## matrixTF
        TFClassical = pd.DataFrame(countTerms.tolist(), columns=listOfTermsTF)
        ### save in file
        logger.info("saving TF File: "+path_for_filesaved+"/tf_on_"+locality.replace("/", "_")+"_corpus.csv")
        TFClassical.to_csv(path_for_filesaved+"/tf_on_"+locality.replace("/", "_")+"_corpus.csv")
        ## Extract N TOP ranking score
        extractBiggestTF = TFClassical.max().nlargest(nb_biggest_terms)
        extractBiggestTF = extractBiggestTF.to_frame()
        extractBiggestTF = extractBiggestTF.reset_index()
        extractBiggestTF.columns = ['terms', 'score']
        extractBiggestTF[spatial_hiearchy] = locality
        extractBiggestTF_allstates = extractBiggestTF_allstates.append(extractBiggestTF, ignore_index=True)
    """

    logger.info("saving TF and TF-IDF top"+str(nb_biggest_terms)+" biggest score")
    extractBiggestTF_allstates.to_csv(path_for_filesaved+"/TF_BiggestScore_on_"+spatial_hiearchy+"_corpus.csv")
    extractBiggestTFIDF_allstates.to_csv(path_for_filesaved+"/TF-IDF_BiggestScore_on_"+spatial_hiearchy+"_corpus.csv")

def TFIDF_TF_on_whole_corpus(elastic_query_fname, logger, path_for_filesaved="./",
                             temporal_period='all', listOfCities='all'):
    """
    Compute TFIDF and TF from an elastic query file
    1 doc = 1 tweet
    Corpus = on the whole elastic query (with filter out cities that are not in listOfCities

    :param elastic_query_fname: filename and path of the elastic query
    :param logger: logger of the main program
    :param nb_biggest_terms: How many biggest term are to keep. It has to be greater than H-TF-IDF or
        TF-IDF classical on corpus = localité because a lot of temrs have 1.0 has the score
    :param spatial_hiearchy: define the size of the corpus : state or country
    :param temporal_period:
    :param listOfCities: If you want to filter out some cities, you can
    :return:
    """
    # tfidfStartDate = date(2020, 1, 23)
    # tfidfEndDate = date(2020, 1, 30)
    # temporal_period = pd.date_range(tfidfStartDate, tfidfEndDate)
    # listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']
    # listOfState = ["England", "Scotland", "Northern Ireland", "Wales"]

    # Query Elasticsearch to get all tweets from UK
    tweets = elasticsearchQuery(elastic_query_fname, logger)
    if listOfCities == 'all':
        listOfCities = []
        listOfStates = []
        listOfCountry = []
        for triple in tweets:
            splitted = triple.split("_")
            listOfCities.append(splitted[0])
            listOfStates.append(splitted[1])
            listOfCountry.append(splitted[2])
        listOfCities = list(set(listOfCities))
        listOfStates = list(set(listOfStates))
        listOfCountry = list(set(listOfCountry))
    # reorganie tweets (dict : tweets by cities) into dataframe (city and date)
    matrixAllTweets = pd.DataFrame()
    for tweetByCity in tweets.keys():
        # Filter cities :
        city = str(tweetByCity).split("_")[0]
        if city in listOfCities:
            matrix = pd.DataFrame(tweets[tweetByCity])
            matrixAllTweets = matrixAllTweets.append(matrix, ignore_index=True)

    # Split datetime into date and time
    matrixAllTweets["date"] = [d.date() for d in matrixAllTweets['created_at']]
    matrixAllTweets["time"] = [d.time() for d in matrixAllTweets['created_at']]

    # Filter by a period
    if temporal_period != "all":
        mask = ((matrixAllTweets["date"] >= temporal_period.min()) & (matrixAllTweets["date"] <= temporal_period.max()))
        matrixAllTweets = matrixAllTweets.loc[mask]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=0.001,
        ngram_range=(1, 2),
        token_pattern='[a-zA-Z0-9#]+', #remove user name, i.e term starting with @ for personnal data issue
    )
    # logger.info("Compute TF-IDF on corpus = "+spatial_hiearchy)
    try:
        vectors = vectorizer.fit_transform(matrixAllTweets['tweet'])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
    except:
        logger.info("Impossible to compute TF-IDF")
        exit(-1)
    ## matrixTFIDF
    TFIDFClassical = pd.DataFrame(denselist, columns=feature_names)
    logger.info("saving TF-IDF File: "+path_for_filesaved+"/tfidf_on_whole_corpus.csv")
    TFIDFClassical.to_csv(path_for_filesaved+"/tfidf_on_whole_corpus.csv")
    ## Extract N TOP ranking score
    extractBiggest = TFIDFClassical.max()
    extractBiggest = extractBiggest[extractBiggest == 1] # we keep only term with high score TF-IDF, i.e 1.0
    extractBiggest = extractBiggest.to_frame()
    extractBiggest = extractBiggest.reset_index()
    extractBiggest.columns = ['terms', 'score']

    logger.info("saving  TF-IDF top"+str(extractBiggest['terms'].size)+" biggest score")
    extractBiggest.to_csv(path_for_filesaved+"/TFIDF_BiggestScore_on_whole_corpus.csv")


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


def logsetup(log_fname):
    """
    Initiate a logger object :
        - Log in file : collectweets.log
        - also print on screen
    :return: logger object
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(funcName)20s() ::%(message)s')
    now = datetime.now()
    file_handler = RotatingFileHandler(log_fname + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log", 'a', 1000000, 1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    # Only display on screen INFO
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


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

def t_SNE_bert_embedding_visualization(biggest_score, logger, listOfLocalities="all", spatial_hieararchy="country",
                                       plotname="colored by country", paht2save="./"):
    """
    Plot t-SNE representation of terms by country
    ressources:
        + https://colab.research.google.com/drive/1FmREx0O4BDeogldyN74_7Lur5NeiOVye?usp=sharing#scrollTo=Fbq5MAv0jkft
        + https://github.com/UKPLab/sentence-transformers
    :param biggest_score:
    :param listOfLocalities:
    :param spatial_hieararchy:
    :param plotname:
    :param paht2save:
    :return:
    """
    modelSentenceTransformer = SentenceTransformer('distilbert-base-nli-mean-tokens')


    # filter by localities
    for locality in biggest_score[spatial_hieararchy].unique():
        if locality not in listOfLocalities:
            biggest_score = biggest_score.drop(biggest_score[biggest_score[spatial_hieararchy] == locality].index)

    embeddings = modelSentenceTransformer.encode(biggest_score['terms'].to_list(), show_progress_bar=True)
    # embeddings.tofile(paht2save+"/tsne_bert-embeddings_"+plotname+"_matrix-embeddig")

    modelTSNE = TSNE(n_components=2)  # n_components means the lower dimension
    low_dim_data = modelTSNE.fit_transform(embeddings)

    label_tsne = biggest_score[spatial_hieararchy]

    # Style Plots a bit
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

    plt.rcParams['figure.figsize'] = (20, 14)

    tsne_df = pd.DataFrame(low_dim_data, label_tsne)
    tsne_df.columns = ['x', 'y']
    ax = sns.scatterplot(data=tsne_df, x='x', y='y', hue=tsne_df.index)
    plt.ylim(-100,100)
    plt.xlim(-100, 100)
    ax.set_title('T-SNE BERT Sentence Embeddings for '+plotname)
    plt.savefig(paht2save+"/tsne_bert-embeddings_"+plotname)
    logger.info("file: "+paht2save+"/tsne_bert-embeddings_"+plotname+" has been saved.")
    #plt.show()
    plt.close()

    # Perform kmean clustering
    # num_clusters = 5
    # clustering_model = KMeans(n_clusters=num_clusters)
    # clustering_model.fit(embeddings)
    # cluster_assignment = clustering_model.labels_

    # Normalize the embeddings to unit length
    corpus_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    # clustered_sentences = [[] for i in range(num_clusters)]
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     clustered_sentences[cluster_id].append(biggest_score['terms'].iloc[sentence_id])

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(biggest_score['terms'].iloc[sentence_id])

    #for i, cluster in enumerate(clustered_sentences):
    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        print(cluster)
        print("")

def bert_embedding_filtred(biggest_score, listOfLocalities="all", spatial_hieararchy="country"):
    """
    Retrieve embedding of a matrix of termes (possibility of filtring by a list of locality)
    :param biggest_score: pd.Datraframe with columns : [terms, country/state/city]
    :param listOfLocalities:
    :param spatial_hieararchy:
    :return:
    """
    modelSentenceTransformer = SentenceTransformer('distilbert-base-nli-mean-tokens')
    # filter by localities
    if listOfLocalities != "all":
        for locality in biggest_score[spatial_hieararchy].unique():
            if locality not in listOfLocalities:
                biggest_score = biggest_score.drop(biggest_score[biggest_score[spatial_hieararchy] == locality].index)

    embeddings = modelSentenceTransformer.encode(biggest_score['terms'].to_list(), show_progress_bar=True)
    return embeddings

def similarity_intra_matrix_pairwise(matrix):
    """
    Compute pairwise cosine similarity on the rows of a Matrix and retrieve unique score by pair.
    indeed, cosine_similarity pairwise retrive a matrix with duplication : let's take an exemple :
    Number of terms : 4, cosine similarity :
            w1   w2  w3  w4
            +---+---+----+--+
        w1  | 1 |   |   |   |
        w2  |   | 1 |   |   |
        w3  |   |   | 1 |   |
        w4  |   |   |   | 1 |
            +---+---+----+--+

        (w1, w2) = (w2, w1), so we have to keep only  : (number_of_terms)^2/2 - (number_of_terms)/2
                                                        for nb_term = 4 :
                                                        4*4/2 - 4/2 = 16/2 - 4/2 = 6 => we have 6 unique scores

    :param matrix:
    :return: list of unique similarity score
    """
    similarity = cosine_similarity(sparse.csr_matrix(matrix))
    similarity_1D = np.array([])
    for i, row in enumerate(similarity):
        similarity_1D = np.append(similarity_1D, row[i+1:]) # We remove duplicate pairwise value
    return similarity_1D

def similarity_inter_matrix(matrix1, matrix2):
    """

    :param matrix1:
    :param matrix2:
    :return:
    """
    similarity = 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')
    return similarity

def geocoding_token(biggest, listOfLocality, spatial_hieararchy, logger):
    """
    Find and geocode Spatial entity with OSM data (nominatim)
    Respect terms and use of OSM and Nomitim :
        - Specify a name for the application, Ie.e user agent
        - add delay between each query : min_delay_seconds = 1.
            See : https://geopy.readthedocs.io/en/stable/#module-geopy.extra.rate_limiter
        - define a time out for waiting nomatim answer : to 10 seconds
    :param biggest:
    :return: biggest with geocoding information
    """
    try:
        if listOfLocality != "all":
            for locality in biggest[spatial_hieararchy].unique():
                if locality not in listOfLocality:
                    biggest = biggest.drop(biggest[biggest[spatial_hieararchy] == locality].index)
    except:
        logger.info("could not filter, certainly because there is no spatial hiearchy on biggest score")

    geolocator = Nominatim(user_agent="h-tfidf-evaluation", timeout=10)
    geocoder = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    tqdm.pandas()
    biggest["geocode"] = biggest["terms"].progress_apply(geocoder)
    return biggest

def post_traitement_flood(biggest, logger, spatialLevel, ratio_of_flood=0.5):
    """
    Remove terms from people flooding : return same dataframe with 1 more column : user_flooding
    With default ratio_of_flood : If an twitter.user use a term in more than 50% of occurence of this terms,
       we consider this user is flooding

    :param biggest: File of terms to process
    :param logger:
    :param: spatialLevel : work on Country / State / City
    :param: ratio_of_flood
    :return: return same dataframe with 1 more column : user_flooding
    """
    ratio_of_flood_global = ratio_of_flood
    es_logger.setLevel(logging.WARNING)

    # pre-build elastic query for spatialLevel :
    rest_user_osm_level = ""
    if spatialLevel == "country":
        rest_user_osm_level = "rest_user_osm.country"
    elif spatialLevel == "state":
        rest_user_osm_level = "rest.features.properties.state"
    elif spatialLevel == "city":
        rest_user_osm_level = "rest.features.properties.city"

    def is_an_user_flooding(term, locality):
        client = Elasticsearch("http://localhost:9200")
        index = "twitter"
        # Query :
        ## Retrieve only user name where in full_text = term and rest_user_osm.country = locality
        if term is not np.NAN:
            query = {"_source": "user.name","query":{"bool":{"filter":[{"bool":{"should":[{"match_phrase":{"full_text":term}}],"minimum_should_match":1}},
                                                                       {"bool":{"should":[{"match_phrase":{rest_user_osm_level:locality}}],"minimum_should_match":1}}]}}}
            try:
                result = Elasticsearch.search(client, index=index, body=query)
                list_of_user = []
                for hit in result["hits"]["hits"]:
                    user = hit["_source"]["user"]["name"]
                    list_of_user.append(user)
                dict_user_nbtweet = dict(Counter(list_of_user))
                d = dict((k, v) for k, v in dict_user_nbtweet.items() if v >= (ratio_of_flood_global * len(list_of_user)))
                if len(d) > 0 : # there is a flood on this term:
                    return 1
                else:
                    return 0
            except:
                logger.debug("Elasticsearch deamon may not be launched or there is a trouble with this term: " + str(term))
                return 0
        else:
            return 0

    logger.debug("start remove terms if they coming from a flooding user, ie, terms in "+str(ratio_of_flood_global*100)+"% of tweets from an unique user over tweets with this words")
    tqdm.pandas()
    biggest["user_flooding"] = biggest.progress_apply(lambda t: is_an_user_flooding(t.terms, t[spatialLevel]), axis=1)
    return biggestr



if __name__ == '__main__':
    # Workflow parameters :
    ## Rebuild H-TFIDF (with Matrix Occurence)
    build_htfidf = True
    ## eval 1 : Comparison with classical TF-IDf
    build_classical_tfidf = True
    ## evla 2 : Use word_embedding with t-SNE
    build_tsne = False
    build_tsne_spatial_level = "country"
    ## eval 3 : Use word_embedding with box plot to show disparity
    build_boxplot = True
    build_boxplot_spatial_level = "country"
    ## post-traitement 1 : geocode term
    build_posttraitement_geocode = False
    ## post-trautement 2 : remove terms form a flooding user
    build_posttraitement_flooding = True
    build_posttraitement_flooding_spatial_levels = ['country', 'state', 'city']

    # Global parameters :
    ## Path to results :
    f_path_result = "elasticsearch/analyse/nldb21/results/4thfeb"
    ## Spatial level hierarchie :
    spatialLevels = ['country', 'state', 'city']
    ## Time level hierarchie :
    timeLevel = "week"
    ## List of country to work on :
    listOfLocalities = ["France", "Deutschland", "España", "Italia", "United Kingdom"]
    ## elastic query :
    query_fname = "elasticsearch/analyse/nldb21/elastic-query/nldb21_europeBySpatialExtent_en_4thweekFeb.txt"

    # initialize a logger :
    log_fname = "elasticsearch/analyse/nldb21/logs/nldb21_"
    logger = logsetup(log_fname)
    logger.info("H-TFIDF expirements starts")


    if build_htfidf:
        # start the elastic query
        query = open(query_fname, "r").read()
        logger.debug("elasticsearch : start quering")
        tweetsByCityAndDate = elasticsearchQuery(query_fname, logger)
        logger.debug("elasticsearch : stop quering")

        # Build a matrix of occurence for each terms in document aggregate by city and day
        ## prepare tree for file in commun for all spatial level :
        f_path_result_common = f_path_result+"/common"
        if not os.path.exists(f_path_result_common):
            os.makedirs(f_path_result_common)
        ## Define file path
        matrixAggDay_fpath = f_path_result_common + "/matrixAggDay.csv"
        matrixOccurence_fpath = f_path_result_common + "/matrixOccurence.csv"
        logger.debug("Build matrix of occurence : start")
        matrixOccurence = matrixOccurenceBuilder(tweetsByCityAndDate, matrixAggDay_fpath, matrixOccurence_fpath, logger)
        logger.debug("Build matrix of occurence : stop")
        ## import matrixOccurence if you don't want to re-build it
        # matrixOccurence = pd.read_csv('elasticsearch/analyse/matrixOccurence.csv', index_col=0)

        for spatialLevel in spatialLevels:
            logger.info("H-TFIDF on: "+spatialLevel)
            f_path_result_level = f_path_result+"/"+spatialLevel
            if not os.path.exists(f_path_result_level):
                os.makedirs(f_path_result_level)
            ## Compute H-TFIDF
            matrixHTFIDF_fname = f_path_result_level + "/matrix_H-TFIDF.csv"
            biggestHTFIDFscore_fname = f_path_result_level + "/h-tfidf-Biggest-score.csv"
            logger.debug("H-TFIDF : start to compute")
            HTFIDF(matrixOcc=matrixOccurence,
                   matrixHTFIDF_fname=matrixHTFIDF_fname,
                   biggestHTFIDFscore_fname=biggestHTFIDFscore_fname,
                   spatialLevel=spatialLevel,
                   temporalLevel=timeLevel,
                   )
        logger.info("H-TFIDF : stop to compute for all spatial levels")

    ## Comparison with TF-IDF
    f_path_result_tfidf = f_path_result + "/tf-idf-classical"
    f_path_result_tfidf_by_locality = f_path_result_tfidf + "/tfidf-tf-corpus-country"
    if build_classical_tfidf :
        if not os.path.exists(f_path_result_tfidf):
            os.makedirs(f_path_result_tfidf)
        if not os.path.exists(f_path_result_tfidf_by_locality):
            os.makedirs(f_path_result_tfidf_by_locality)
        ### On whole corpus
        TFIDF_TF_on_whole_corpus(elastic_query_fname=query_fname,
                                 logger=logger,
                                 path_for_filesaved=f_path_result_tfidf)
        ### By Country
        TFIDF_TF_with_corpus_state(elastic_query_fname=query_fname,
                                   logger=logger,
                                   nb_biggest_terms=500,
                                   path_for_filesaved=f_path_result_tfidf_by_locality,
                                   spatial_hiearchy="country",
                                   temporal_period='all')
    if build_tsne :
        f_path_result_tsne = f_path_result+"/tsne"
        biggest_TFIDF_country = pd.read_csv(f_path_result_tfidf_by_locality+"TF-IDF_BiggestScore_on_country_corpus.csv", index_col=0)
        biggest_TFIDF_whole = pd.read_csv(f_path_result_tfidf+"TFIDF_BiggestScore_on_whole_corpus.csv")
        biggest_H_TFIDF = pd.read_csv(f_path_result+"/"+build_tsne_spatial_level+'/h-tfidf-Biggest-score.csv', index_col=0)
        # t_SNE visulation
        t_SNE_bert_embedding_visualization(biggest_TFIDF_country, logger, listOfLocalities=listOfLocalities,
                                           plotname="TF-IDF on corpus by Country",
                                           paht2save=f_path_result_tsne)
        t_SNE_bert_embedding_visualization(biggest_H_TFIDF, logger, listOfLocalities=listOfLocalities,
                                           plotname="H-TFIDF", paht2save=f_path_result_tsne)

    if build_boxplot :
        # dir path to save :
        f_path_result_boxplot = f_path_result+"/pairwise-similarity-boxplot"
        if not os.path.exists(f_path_result_boxplot):
            os.makedirs(f_path_result_boxplot)
        # open result from mesures :
        biggest_TFIDF_country = pd.read_csv(f_path_result_tfidf_by_locality+"/TF-IDF_BiggestScore_on_country_corpus.csv", index_col=0)
        biggest_TFIDF_whole = pd.read_csv(f_path_result_tfidf+"/TFIDF_BiggestScore_on_whole_corpus.csv")
        biggest_H_TFIDF = pd.read_csv(f_path_result+"/"+build_boxplot_spatial_level+'/h-tfidf-Biggest-score.csv', index_col=0)
        # Retrieve embedding :
        htfidf_embeddings = bert_embedding_filtred(biggest_H_TFIDF, listOfLocalities=listOfLocalities)
        tfidf_country_embeddings = bert_embedding_filtred(biggest_TFIDF_country, listOfLocalities=listOfLocalities)
        tfidf_whole_embeddings = bert_embedding_filtred(biggest_TFIDF_whole)
        # Compute similarity :
        ## Distribution of similarities between terms extracted from a measure
        htidf_similarity = similarity_intra_matrix_pairwise(htfidf_embeddings)
        tfidf_country_similarity = similarity_intra_matrix_pairwise(tfidf_country_embeddings)
        tfidf_whole_similarity = similarity_intra_matrix_pairwise(tfidf_whole_embeddings)

        plt.subplot(131)
        plt.boxplot(htidf_similarity)
        plt.title("H-TFIDF")
        plt.ylim(0,1)
        plt.subplot(132)
        plt.boxplot(tfidf_country_similarity)
        plt.title("TFIDF with corpus by country")
        plt.ylim(0, 1)
        plt.subplot(133)
        plt.boxplot(tfidf_whole_similarity)
        plt.title("TFIDF on the whole corpus")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.suptitle("Distribution of similarity values among the extracted terms pairs of a measure")
        plt.savefig(f_path_result_boxplot+"/pairwise-similarity-boxplot.png")
        # plt.show()
        plt.close()
        ## Distribution of similarities between the terms of a country extracted from a measure
        ### H-TFIDF
        fig2, axs2 = plt.subplots(1, 5)
        for i, country in enumerate(listOfLocalities):
            axs2[i].boxplot(similarity_intra_matrix_pairwise(htfidf_embeddings[i*500:(i+1)*500-1]))
            axs2[i].set_title(country)
            axs2[i].set_ylim(0, 1)
        fig2.suptitle("Distribution of similarity by pairs for H-TF-IDF")
        plt.savefig(f_path_result_boxplot + "/pairwise-similarity-boxplot_HTFIDF-country.png")
        # plt.show()
        plt.close(fig2)
        ### TF-IDF by corpus = country
        fig3, axs3 = plt.subplots(1, 5)
        for i, country in enumerate(listOfLocalities):
            axs3[i].boxplot(similarity_intra_matrix_pairwise(tfidf_country_embeddings[i*500:(i+1)*500-1]))
            axs3[i].set_title(country)
            axs3[i].set_ylim(0, 1)
        fig3.suptitle("Distribution of similarity by pairs for TF-IDF focus on each country")
        plt.savefig(f_path_result_boxplot + "/pairwise-similarity-boxplot_TFIDF-country.png")
        # plt.show()
        plt.close(fig3)
        ## Distribution of similarities between the set of terms of 2 measures
        ### H-TF-IDF with TF-IDF on whole corpus and TF-IDF country with TF-IDF on whole corpus
        fig_compare_TFIDF_whole, ax4 = plt.subplots(1,2)
        similarity_between_htfidf_tfidf_whole = similarity_inter_matrix(htfidf_embeddings, tfidf_whole_embeddings)
        similarity_between_tfidfcountry_tfidf_whole = similarity_inter_matrix(tfidf_country_embeddings, tfidf_whole_embeddings)
        similarity_between_htfidf_tfidf_whole_1D = np.array([])
        similarity_between_tfidfcountry_tfidf_whole_1D = np.array([])
        for i, row in enumerate(similarity_between_htfidf_tfidf_whole):
            similarity_between_htfidf_tfidf_whole_1D = np.append(similarity_between_htfidf_tfidf_whole_1D, row[i+1:]) # We remove duplicate pairwise value
        for i, row in enumerate(similarity_between_tfidfcountry_tfidf_whole):
            similarity_between_tfidfcountry_tfidf_whole_1D = np.append(similarity_between_tfidfcountry_tfidf_whole_1D,
                                                                 row[i + 1:])
        ax4[0].boxplot(similarity_between_htfidf_tfidf_whole_1D)
        ax4[0].set_ylim(0, 1)
        ax4[0].set_title("H-TFIDF")
        ax4[1].boxplot(similarity_between_tfidfcountry_tfidf_whole_1D)
        ax4[1].set_ylim(0, 1)
        ax4[1].set_title("TFIDF on country")
        fig_compare_TFIDF_whole.suptitle("Distribution of similarity between H-TFIDF and TF-IDF on whole corpus")
        plt.savefig(f_path_result_boxplot + "/pairwise-similarity-boxplot_between_TFIDF-whole.png")
        # plt.show()
        plt.close(fig_compare_TFIDF_whole)
        ## Distribution of similarities between sub-set terms by country compared by country pair

    if build_posttraitement_geocode:
        # Geocode terms :
        ## Comments : over geocode even on non spatial entities
        spatial_level = "country"
        listOfLocalities = ["France", "Deutschland", "España", "Italia", "United Kingdom"]
        f_path_result = "elasticsearch/analyse/nldb21/results/4thfeb_country"
        biggest_TFIDF_country = pd.read_csv(
            f_path_result+"/tfidf-tf-corpus-country/TF-IDF_BiggestScore_on_"+spatial_level+"_corpus.csv", index_col=0)
        biggest_TFIDF_whole = pd.read_csv(f_path_result+"/TFIDF_BiggestScore_on_whole_corpus.csv")
        biggest_H_TFIDF = pd.read_csv(f_path_result+'/h-tfidf-Biggest-score.csv', index_col=0)
        biggest_H_TFIDF_gepocode = geocoding_token(biggest_H_TFIDF,
                                                   listOfLocality=listOfLocalities,
                                                   spatial_hieararchy=spatial_level,
                                                   logger=logger)
        biggest_H_TFIDF_gepocode.to_csv(f_path_result+"/h-tfidf-Biggest-score-geocode.csv")
        biggest_TFIDF_country_gepocode = geocoding_token(biggest_TFIDF_country,
                                                   listOfLocality=listOfLocalities,
                                                   spatial_hieararchy=spatial_level,
                                                         logger=logger)
        biggest_TFIDF_country_gepocode.to_csv(f_path_result+"/TF-IDF_BiggestScore_on_"+spatial_level+"_corpus_geocode.csv")
        biggest_TFIDF_whole_gepocode = geocoding_token(biggest_TFIDF_whole,
                                                   listOfLocality=listOfLocalities,
                                                   spatial_hieararchy=spatial_level,
                                                       logger=logger)
        biggest_TFIDF_whole_gepocode.to_csv(f_path_result+"/TFIDF_BiggestScore_on_whole_corpus_geocode.csv")


    if build_posttraitement_flooding:
        # Post traitement : remove terms coming from user who flood
        for spatial_level_flood in build_posttraitement_flooding_spatial_levels:
            logger.info("post-traitement flooding on: " + spatialLevel)
            f_path_result_flood = f_path_result + "/" + spatialLevel
            biggest_H_TFIDF = pd.read_csv(f_path_result_flood + '/h-tfidf-Biggest-score.csv', index_col=0)
            biggest_H_TFIDF_with_flood = post_traitement_flood(biggest_H_TFIDF, logger, spatialLevel=spatialLevel)
            biggest_H_TFIDF_with_flood.to_csv(f_path_result_flood + "/h-tfidf-Biggest-score-flooding.csv")

    logger.info("H-TFIDF expirements stops")
