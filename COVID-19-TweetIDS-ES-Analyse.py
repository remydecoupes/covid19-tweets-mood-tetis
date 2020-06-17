#!/usr/bin/env python

"""
Stat on dataset
"""
import json
from elasticsearch import Elasticsearch, exceptions

if __name__ == '__main__':
    print("begin")
    client = Elasticsearch("http://localhost:9200")
    index = "twitter"
    query = { "query": {    "bool": {      "must": [        {          "match": {            "rest_user_osm.country.keyword": "United Kingdom"          }        },        {          "range": {            "created_at": {              "gte": "Wed Jan 22 00:00:01 +0000 2020"            }          }        }      ]    }  }}
    result = Elasticsearch.search(client, index=index, body=query)
    # print(json.dumps(result["hits"]["hits"], indent=4))
    for hits in result["hits"]["hits"]:
        # if city properties is available on OSM
        print(json.dumps(hits["_source"]["rest"]["features"][0]["properties"], indent=4))
        if "city" in hits["_source"]["rest"]["features"][0]["properties"]:
            print(json.dumps(hits["_source"]["rest"]["features"][0]["properties"]["city"], indent=4))
    print("end")