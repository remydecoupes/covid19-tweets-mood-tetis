#!/usr/bin/env python

"""
analyse Elasticsearch query
"""

from elasticsearch import Elasticsearch
from collections import defaultdict
import re
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

# Global var on Levels on spatial and temporal axis
spatialLevels = ['city', 'state', 'country']
temporalLevels = ['day', 'week', 'month', 'period']


def elasticsearchQuery():
    """
    Build a ES query  and return a default dict with resuls
    :return: tweetsByCityAndDate
    """
    # Elastic search credentials
    client = Elasticsearch("http://localhost:9200")
    index = "twitter"
    # Define a Query : Here get only city from UK
    query = {"query": {"bool": {"must": [{"match": {"rest_user_osm.country.keyword": "United Kingdom"}},
                                         {"range": {"created_at": {"gte": "Wed Jan 22 00:00:01 +0000 2020"}}}]}}}
    result = Elasticsearch.search(client, index=index, body=query, scroll='5m')

    # Append all pages form scroll search : avoid the 10k limitation of ElasticSearch
    results = avoid10kquerylimitation(result, client)

    # Initiate a dict for each city append all Tweets content
    tweetsByCityAndDate = defaultdict(list)
    for hits in results:
        # if city properties is available on OSM
        # print(json.dumps(hits["_source"]["rest"]["features"][0]["properties"], indent=4))
        if "city" in hits["_source"]["rest"]["features"][0]["properties"]:
            # parse Java date : EEE MMM dd HH:mm:ss Z yyyy
            inDate = hits["_source"]["created_at"]
            parseDate = datetime.strptime(inDate, "%a %b %d %H:%M:%S %z %Y")
            cityStateCountry = str(hits["_source"]["rest"]["features"][0]["properties"]["city"]) + "_" + \
                               str(hits["_source"]["rest"]["features"][0]["properties"]["state"]) + "_" + \
                               str(hits["_source"]["rest"]["features"][0]["properties"]["country"])
            tweetsByCityAndDate[cityStateCountry].append(
                {
                    "tweet": preprocessTweets(hits["_source"]["full_text"]),
                    "created_at": parseDate
                }
            )
    # biotexInputBuilder(tweetsByCityAndDate)
    # pprint(tweetsByCityAndDate)
    return tweetsByCityAndDate


def avoid10kquerylimitation(result, client):
    """
    Elasticsearch limit results of query at 10 000. To avoid this limit, we need to paginate results and scroll
    This method append all pages form scroll search
    :param result: a result of a ElasticSearcg query
    :return:
    """
    scroll_size = result['hits']['total']["value"]
    results = []
    while (scroll_size > 0):
        try:
            scroll_id = result['_scroll_id']
            res = client.scroll(scroll_id=scroll_id, scroll='60s')
            results += res['hits']['hits']
            scroll_size = len(res['hits']['hits'])
        except:
            break
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
    textclean = re.sub('@[^\s]+', '', textclean)
    # remove the # in #hashtag
    textclean = re.sub(r'#([^\s]+)', r'\1', textclean)
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


def matrixOccurenceBuilder(tweetsofcity):
    """
    Create a matrix of :
        - line : (city,day)
        - column : terms
        - value of cells : TF (term frequency)
    Help found here :
    http://www.xavierdupre.fr/app/papierstat/helpsphinx/notebooks/artificiel_tokenize_features.html
    https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    :param tweetsofcity:
    :return:
    """
    # initiate matrix of tweets aggregate by day
    # col = ['city', 'day', 'tweetsList', 'bow']
    col = ['city', 'day', 'tweetsList']
    matrixAggDay = pd.DataFrame(columns=col)
    cityDayList = []

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
            document = '\n'.join(matrix.loc[matrix['created_at'].dt.date == day]['tweet'].tolist())
            # Bag of Words and preprocces
            # bagOfWords = preprocessTerms(document).split(" ")
            tweetsOfDayAndCity = {
                'city': city,
                'day': day,
                'tweetsList': document,
                #    'bow': bagOfWords
            }
            cityDayList.append(city + "_" + str(day))
            matrixAggDay = matrixAggDay.append(tweetsOfDayAndCity, ignore_index=True)
    matrixAggDay.to_csv("elasticsearch/analyse/matrixAggDay.csv")

    # Count terms with sci-kit learn
    cd = CountVectorizer()
    cd.fit(matrixAggDay['tweetsList'])
    res = cd.transform(matrixAggDay["tweetsList"])
    countTerms = res.todense()
    # create matrix
    ## get terms :
    # voc = cd.vocabulary_
    # listOfTerms = {term for term, index in sorted(voc.items(), key=lambda item: item[1])}
    listOfTerms = cd.get_feature_names()
    ## initiate matrix with count for each terms
    matrixOccurence = pd.DataFrame(data=countTerms[0:, 0:], index=cityDayList, columns=listOfTerms)
    ## Remove stopword
    for term in matrixOccurence.keys():
        if term in stopwords.words('english'):
            del matrixOccurence[term]

    # save to file
    matrixOccurence.to_csv("elasticsearch/analyse/matrixOccurence.csv")
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


def TFIDFAdaptative(matrixOcc, listOfcities='all', spatialLevel='city', period='all', temporalLevel='day'):
    """
    Aggregate on spatial and temporel and then compute TF-IDF

    :param matrixOcc: Matrix with TF already compute
    :param listOfcities: filter on this cities
    :param spatialLevel: city / state / country / world
    :param period: Filter on this period
    :param temporalLevel: TBD
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
    ## In space
    if spatialLevel == 'city':
        # do nothing
        pass
    elif spatialLevel == 'state':
        matrixOcc = matrixOcc.groupby("state").sum()
    elif spatialLevel == 'country':
        matrixOcc = matrixOcc.groupby("country").sum()

    # Compute TF-IDF
    ## compute TF : for each doc, devide count by Sum of all count
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
    matrixTFIDF.to_csv("elasticsearch/analyse/matrixTFIDFadaptative.csv")

    # Export N biggest TF-IDF score:
    top_n = 500
    extractBiggest = pd.DataFrame(index=matrixTFIDF.index, columns=range(0, top_n))
    for row in matrixTFIDF.index:
        extractBiggest.loc[row] = matrixTFIDF.loc[row].nlargest(top_n).keys()
    extractBiggest.to_csv("elasticsearch/analyse/TFIDFadaptativeBiggestScore.csv")


def ldHTFIDFadaptative(listOfcities):
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
    .. warnings:: /!\ under dev !!!. See TODO below
    .. todo::
        - Remove filter and pass it as args :
            - period
            - list of Cities
        - Pass files path in args
        - Pass number of term to extract for TF-IDF and TF
    Gives commons and specifics terms between H-TFIDF and TF & TF-IDF classics
    Creates 6 csv files : 3 for earch classical measures :
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

    #  Filter by a period
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


if __name__ == '__main__':
    print("begin")
    """
    # Comment below if you don't want to rebuild matrixOccurence
    # Query Elastic Search : From now only on UK (see functions var below)
    tweetsByCityAndDate = elasticsearchQuery()
    # Build a matrix of occurence for each terms in document aggregate by city and day
    matrixOccurence = matrixOccurenceBuilder(tweetsByCityAndDate)
    """
    # TF-IDF adaptative
    ## import matrixOccurence if you don't want to re-build it
    """
    # matrixOccurence = pd.read_csv('elasticsearch/analyse/matrixOccurence.csv', index_col=0)
    """
    ### Filter city and period
    listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']
    tfidfStartDate = date(2020, 1, 23)
    tfidfEndDate = date(2020, 1, 30)
    tfidfPeriod = pd.date_range(tfidfStartDate, tfidfEndDate)

    """
    ## Compute TF-IDF
    TFIDFAdaptative(matrixOcc=matrixOccurence, listOfcities=listOfCity, spatialLevel='state', period=tfidfPeriod)

    # LDA clustering on TF-IDF adaptative vocabulary
    listOfCityState = ['London_England', 'Glasgow_Scotland', 'Belfast_Northern Ireland', 'Cardiff_Wales']
    ldHTFIDFadaptative(listOfCityState)
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
    tfidfpath = "elasticsearch/analyse/TFIDFClassical/TFIDFclassicalBiggestScore.csv"
    tfpath = "elasticsearch/analyse/TFClassical/TFclassicalBiggestScore.csv"

    """
    #Compare classical TF-IDF with H-TFIDF
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
    tfidf = pd.read_csv(tfidfpath)
    tf = pd.read_csv(tfpath)
    htfidfStackedPAth = "elasticsearch/analyse/h-tfidf-stacked-wordnet.csv"
    htfidf = pd.read_csv(htfidfStackedPAth)
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
    #Point 6 Comment thesaurus coverage
    
    ## plot graph coverage depending nb first elements
    ### Retrieve the mimimun len (i.e. nb of terms extracted) for the three measure :
    min_len = min(tfidf.shape[0], tf.shape[0], htfidf.shape[0])

    ### Building dataframes containing percent of thesaurus coverage to plot
    nbfirstelementsRange = range(1,min_len)
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
    wordnetCoverageByNbofterms.plot(x='Number_of_the_first_terms_extracted', y=['h-tfidf', 'tf-idf', 'tf'], kind='line', ax=axes[0])
    axes[1].set(
        xlabel='Number of the first n elements',
        ylabel='Percentage of terms in Agrovoc',
        title='Agrovoc'
    )
    axes[1].xaxis.set_visible(False)
    agrovocCoverageByBbofterms.plot(x='Number_of_the_first_terms_extracted', y=['h-tfidf', 'tf-idf', 'tf'], kind='line', ax=axes[1])
    axes[2].set(
        xlabel='Number of the first n elements',
        ylabel='Percentage of terms in MeSH',
        title='MeSH'
    )
    #axes[2].xaxis.set_visible(False)
    meshCoverageByBbofterms.plot(x='Number_of_the_first_terms_extracted', y=['h-tfidf', 'tf-idf', 'tf'], kind='line', ax=axes[2])
    # As we hide xlabel for each subplots, we want to share one xlabel below the figure
    # fig.text(0.32, 0.04, "Number of the first n elements")
    fig.suptitle("Percentage of terms in Wordnet / Agrovoc / MesH \nby measures H-TFIDF / TF-IDF / TF")
    fig.set_size_inches(8, 15)
    #plt.show()
    #fig.savefig("elasticsearch/analyse/thesaurus_coverage.png")

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
                              #wordcloud_kwargs=dict(max_font_size=10, min_font_size=10),
                              #set_colors=['r', 'g', 'b'],
                              #alpha=0.8,
                              ax=axvenn[2])
    for label in tf_venn.set_labels:
        label.set_fontsize(15)

    plt.show()
    
    # End of thesaurus coverage
    """

    # Point 7 : count  the number of TF / TF-IDF / H-TFIDF terms for each states
    nb_of_extracted_terms_from_mesure = 100
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
        df = htfidf.loc[state].to_frame().set_index(state).join(es_tweets_results_filtred_aggstate.loc[state], how="left")
        df.to_csv("elasticsearch/analyse/state_coverage/htfidf_"+state+".csv")

    # end Point 7

    # Point 8 : Get K frequent terms for each state's tweets and see percentage coverage with the 3 measures
    k_first_terms = 100 # from each state get k first most frequent word
    nb_of_extracted_terms_from_mesure = 100 # from each measure, take nb first terms extract
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
        state_frequent_terms_by_measure.loc[state_frequent_terms_by_measure.state == state, "tf-idf"] =\
            state_frequent_terms_by_measure[state_frequent_terms_by_measure.state == state].join(
            tfidf.iloc[0:nb_of_extracted_terms_from_mesure].set_index("terms"),
            on="terms",
            how='left'
        )["score"]
    ## save in CSV
    state_frequent_terms_by_measure.to_csv("elasticsearch/analyse/state_coverage/eval_point_8.csv")
    ## build barchart
    barchart_col = ["tf", "tf-idf", "h-tfidf"]
    barchart = pd.DataFrame(columns=barchart_col, index=range(1))
    barchart.tf = state_frequent_terms_by_measure.tf.count() / len(state_frequent_terms_by_measure) * 100
    barchart["tf-idf"] = state_frequent_terms_by_measure["tf-idf"].count() / len(state_frequent_terms_by_measure) * 100
    barchart["h-tfidf"] = state_frequent_terms_by_measure["h-tfidf"].count() / len(state_frequent_terms_by_measure) * 100
    barchart = barchart.transpose()
    barchart.plot.bar(title="Percentage of top K first frequent terms presents in measure",
                      legend=False)
    barchart_by_state = state_frequent_terms_by_measure.groupby(["state"]).count()
    barchart_by_state[["tf", "tf-idf", "h-tfidf"]].plot.bar(
        title="Percentage of top K first frequent terms presents in measure by state"
    )
    plt.show()
    # end point 8

    # Point 9 : evaluation with TF / TF-IDF 1 doc = 1 tweet & Corpus = state
    ## Compute TF / TF-IDF by state

    # end point 9


    print("end")
