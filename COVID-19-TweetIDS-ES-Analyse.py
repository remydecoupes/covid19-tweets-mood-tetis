#!/usr/bin/env python

"""
analyse Elasticsearch query
"""
import json
from pprint import pprint
from elasticsearch import Elasticsearch, exceptions
from collections import defaultdict
import re
from pathlib import Path
from datetime import datetime, date, timedelta
# Preprocess terms for TF-IDF
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from num2words import num2words
# end of preprocess
# LDA
from gensim import corpora, models
import pyLDAvis.gensim
## print in coloer
from termcolor import colored
# end LDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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
    :param tweet:
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


def ldaTFIDFadaptative(listOfcities):
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


if __name__ == '__main__':
    print("begin")
    # Comment below if you don't want to rebuild matrixOccurence
    # Query Elastic Search : From now only on UK (see functions var below)
    tweetsByCityAndDate = elasticsearchQuery()
    # Build a matrix of occurence for each terms in document aggregate by city and day
    matrixOccurence = matrixOccurenceBuilder(tweetsByCityAndDate)
    # TF-IDF adaptative
    ## import matrixOccurence if you don't want to re-build it
    """
    matrixOccurence = pd.read_csv('elasticsearch/analyse/matrixOccurence.csv', index_col=0)
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
    ldaTFIDFadaptative(listOfCityState)
    """

    ## Build biotex input for adaptative level state
    biotexAdaptativeBuilderAdaptative(listOfcities=listOfCity, spatialLevel='state',
                                      period=tfidfPeriod, temporalLevel='day')

    print("end")
