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

def avoid10kquerylimitation(result):
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


def matrixTFBuilder(tweetsofcity):
    """
    Create a matrix of :
        - line : (city,day)
        - column : terms
        - value of cells : TF (term frequency)
    Help found here :
    https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    :param tweetsofcity:
    :return:
    """
    # initiate matrix of tweets aggregate by day
    col = ['city', 'day', 'tweetsList', 'bow']
    matrixAggDay = pd.DataFrame(columns=col)

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
            bagOfWords = preprocessTerms(document).split(" ")
            tweetsOfDayAndCity = {
                'city': city,
                'day': day,
                'tweetsList': document,
                'bow': bagOfWords
            }
            matrixAggDay = matrixAggDay.append(tweetsOfDayAndCity, ignore_index=True)
    matrixAggDay.to_csv("elasticsearch/analyse/matrixAggDay.csv")

    # Create Term matrix
    ## Find unique word :
    uniqueTerms = ()
    i = 0
    for cityday in matrixAggDay['bow']:
        if i > 0:
            # uniqueTerms = list(set(cityday.strip('][').split(', ')) | set(uniqueTerms))
            uniqueTerms = list(set(cityday) | set(uniqueTerms))
        else:
            # For 1rst document
            # tip : strip : convert string into list because BoW are in type string  when import with pd.read_csv() :
            # uniqueTerms = cityday.strip('][').split(', ')
            uniqueTerms = cityday
        i += 1
    uniqueTerms.sort()
    ## create matrix
    col = ['city_day']
    col.extend(uniqueTerms)
    matrixTF = pd.DataFrame(columns=col)
    for index, cityday in matrixAggDay.iterrows():
        numOfWords = dict.fromkeys(uniqueTerms, 0)
        print(type(cityday['bow']))
        for word in cityday['bow']:
            numOfWords[word] += 1
        row = [str(cityday['city'])+"_"+str(cityday['day'])]
        row.extend(numOfWords.values())
        matrixTF.append(row)
    matrixTF.to_csv("elasticsearch/analyse/matrixTF.csv")




if __name__ == '__main__':
    print("begin")
    # Elastic search credentials
    client = Elasticsearch("http://localhost:9200")
    index = "twitter"
    # Define a Query : Here get only city from UK
    query = { "query": {    "bool": {      "must": [        {          "match": {            "rest_user_osm.country.keyword": "United Kingdom"          }        },        {          "range": {            "created_at": {              "gte": "Wed Jan 22 00:00:01 +0000 2020"            }          }        }      ]    }  }}
    result = Elasticsearch.search(client, index=index, body=query, scroll='5m')

    # Append all pages form scroll search : avoid the 10k limitation of ElasticSearch
    results = avoid10kquerylimitation(result)

    # Initiate a dict for each city append all Tweets content
    tweetsByCityAndDate = defaultdict(list)
    for hits in results:
        # if city properties is available on OSM
        #print(json.dumps(hits["_source"]["rest"]["features"][0]["properties"], indent=4))
        if "city" in hits["_source"]["rest"]["features"][0]["properties"]:
            # parse Java date : EEE MMM dd HH:mm:ss Z yyyy
            inDate = hits["_source"]["created_at"]
            parseDate = datetime.strptime(inDate, "%a %b %d %H:%M:%S %z %Y")
            tweetsByCityAndDate[hits["_source"]["rest"]["features"][0]["properties"]["city"]].append(
                {
                    "tweet": preprocessTweets(hits["_source"]["full_text"]),
                    "created_at": parseDate
                }
            )
    # biotexInputBuilder(tweetsByCityAndDate)
    # pprint(tweetsByCityAndDate)
    matrixTFBuilder(tweetsByCityAndDate)
    print("end")