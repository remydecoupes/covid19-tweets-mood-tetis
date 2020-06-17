#!/usr/bin/env python

"""
analyse Elasticsearch query
"""
import json
from elasticsearch import Elasticsearch, exceptions
from collections import defaultdict
import re
from pathlib import Path

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
    biotexcorpus = []
    for city in tweetsofcity :
        print("Build document for city: "+city)
        document = tweetsofcity[city]
        # print(document)
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

if __name__ == '__main__':
    print("begin")
    client = Elasticsearch("http://localhost:9200")
    index = "twitter"
    # Define a Query : Here get only city from UK
    query = { "query": {    "bool": {      "must": [        {          "match": {            "rest_user_osm.country.keyword": "United Kingdom"          }        },        {          "range": {            "created_at": {              "gte": "Wed Jan 22 00:00:01 +0000 2020"            }          }        }      ]    }  }}
    result = Elasticsearch.search(client, index=index, body=query, scroll='5m')

    # Append all pages form scroll search : avoid the 10k limitation of ElasticSearch
    results = avoid10kquerylimitation(result)

    # Initiate a dict for each city append all Tweets content
    tweetsByCity = defaultdict()
    for hits in results:
        # if city properties is available on OSM
        #print(json.dumps(hits["_source"]["rest"]["features"][0]["properties"], indent=4))
        if "city" in hits["_source"]["rest"]["features"][0]["properties"]:
            # print city :
            # print(json.dumps(hits["_source"]["rest"]["features"][0]["properties"]["city"], indent=4))
            tweetsByCity[hits["_source"]["rest"]["features"][0]["properties"]["city"]] = \
                    preprocessTweets(hits["_source"]["full_text"])
    biotexInputBuilder(tweetsByCity)
    print("end")