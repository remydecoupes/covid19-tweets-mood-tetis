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
#end of preprocess
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Global var on Levels on spatial and temporal axis
spatialLevels = ['city', 'state', 'country']
temporalLevels = ['day', 'week','month', 'period']

def elasticsearchQuery():
    # Elastic search credentials
    client = Elasticsearch("http://localhost:9200")
    index = "twitter"
    # Define a Query : Here get only city from UK
    query = { "query": {    "bool": {      "must": [        {          "match": {            "rest_user_osm.country.keyword": "United Kingdom"          }        },        {          "range": {            "created_at": {              "gte": "Wed Jan 22 00:00:01 +0000 2020"            }          }        }      ]    }  }}
    result = Elasticsearch.search(client, index=index, body=query, scroll='5m')

    # Append all pages form scroll search : avoid the 10k limitation of ElasticSearch
    results = avoid10kquerylimitation(result, client)

    # Initiate a dict for each city append all Tweets content
    tweetsByCityAndDate = defaultdict(list)
    for hits in results:
        # if city properties is available on OSM
        #print(json.dumps(hits["_source"]["rest"]["features"][0]["properties"], indent=4))
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
    :param tweetsofcity: dictionnary of { tweets, created_at }
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
    doc = removesinglechar(doc) # apostrophe create new single char
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
    #col = ['city', 'day', 'tweetsList', 'bow']
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
            cityDayList.append(city+"_"+str(day))
            matrixAggDay = matrixAggDay.append(tweetsOfDayAndCity, ignore_index=True)
    matrixAggDay.to_csv("elasticsearch/analyse/matrixAggDay.csv")

    # Count terms with sci-kit learn
    cd = CountVectorizer()
    cd.fit(matrixAggDay['tweetsList'])
    res = cd.transform(matrixAggDay["tweetsList"])
    countTerms = res.todense()
    # create matrix
    ## get terms :
    voc = cd.vocabulary_
    listOfTerms = {term for term, index in sorted(voc.items(), key=lambda item: item[1])}
    ## initiate matrix with count for each terms
    matrixOccurence = pd.DataFrame(data=countTerms[0:, 0:], index=cityDayList, columns=listOfTerms)
    matrixOccurence.to_csv("elasticsearch/analyse/matrixOccurence.csv")
    return matrixOccurence

def TFIDFAdaptative(matrixOcc, listOfcities='all', spatialLevel='city', period='all', temporalLevel='day'):
    if spatialLevel not in spatialLevels or temporalLevel not in temporalLevels:
        print("wrong level, please double check")
        return 1
    # Extract cities and period
    ## cities
    if listOfcities != 'all': ### we need to filter
        ### Initiate a numpy array of False
        filter = np.zeros((1, len(matrixOcc.index)), dtype=bool)[0]
        for city in listOfcities:
            ### edit filter if index contains the city (for each city of the list)
            filter += matrixOcc.index.str.startswith(str(city)+"_")
        matrixOcc = matrixOcc.loc[filter]
    ## period
    if str(period) != 'all': ### we need a filter on date
        datefilter = np.zeros((1, len(matrixOcc.index)), dtype=bool)[0]
        for date in period:
            datefilter += matrixOcc.index.str.contains(date.strftime('%Y-%m-%d'))
        matrixOcc = matrixOcc.loc[datefilter]
    print(matrixOcc)

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
    DFt = matrixOcc.astype(bool).sum(axis=0) # Tip : convert all value in boolean. float O,O will be False, other True
    #### Not a Number when value 0 because otherwise log is infinite
    DFt.replace(0, np.nan, inplace=True)
    ### compute log(N/DFt)
    idf = np.log(N/DFt)
    ## compute TF-IDF
    matrixTFIDF = matrixOcc * idf

    # Save file
    matrixTFIDF.to_csv("elasticsearch/analyse/matrixTFIDFadaptative.csv")



if __name__ == '__main__':
    print("begin")
    # Comment below if you don't want to rebuild matrixOccurence
    """
    # Query Elastic Search : From now only on UK (see functions var below)
    tweetsByCityAndDate = elasticsearchQuery()
    # Build a matrix of occurence for each terms in document aggregate by city and day
    matrixOccurence = matrixOccurenceBuilder(tweetsByCityAndDate)
    """
    # TF-IDF adaptative
    ## import matrixOccurence if you don't want to re-build it
    matrixOccurence = pd.read_csv('elasticsearch/analyse/matrixOccurence.csv', index_col=0)
    ### Filter city and period
    listOfCity = ['London', 'Glasgow', 'Belfast']
    tfidfStartDate = date(2020, 1, 23)
    tfidfEndDate = date(2020, 1, 30)
    tfidfPeriod = pd.date_range(tfidfStartDate, tfidfEndDate)
    TFIDFAdaptative(matrixOcc=matrixOccurence, listOfcities=listOfCity, spatialLevel='state', period=tfidfPeriod)

    print("end")