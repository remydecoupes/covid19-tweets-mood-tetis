# List of usefull queries
You can easily export your visual filter from kibana to a query of elastic search for curl or python.

1. build your filter and see it's correct
2. Click on inspect 
3. Click on request
4. Copy paste the query to kibana dev tool or your curl or python script

![Kibana](kibana_export_query_from_visual.png)
## Get all tweets from United Kingdom 
### Kibana dev tools
```
GET twitter/_search?scroll=1m&pretty
{
  "size": 10000,
  "query": {
    "bool": {
      "must": [
        {
          "match" :{
            "rest_user_osm.country.keyword" : "United Kingdom" 
          }
        },
        {
          "range": {
            "created_at": {
              "gte": "Wed Jan 22 00:00:01 +0000 2020"
            }
          }
        }]
    }
  }
}
```
### cURL syntax :
```
curl -XGET "http://localhost:9200/twitter/_search?scroll=1m&pretty" -H 'Content-Type: application/json' -d'{  "size": 10000,  "query": {    "bool": {      "must": [        {          "match" :{            "rest_user_osm.country.keyword" : "United Kingdom"           }        },        {          "range": {            "created_at": {              "gte": "Wed Jan 22 00:00:01 +0000 2020"            }          }        }]    }  }}'
```

### Get tweets with an extracted term AND with spatial and temporal filter :
```
GET twitter/_search?scroll=1m&pretty
{
  "version": true,
  "size": 500,
  "sort": [
    {
      "created_at": {
        "order": "desc",
        "unmapped_type": "boolean"
      }
    }
  ],
  "aggs": {
    "2": {
      "date_histogram": {
        "field": "created_at",
        "fixed_interval": "3h",
        "time_zone": "Europe/Paris",
        "min_doc_count": 1
      }
    }
  },
  "stored_fields": [
    "*"
  ],
  "script_fields": {},
  "docvalue_fields": [
    {
      "field": "@timestamp",
      "format": "date_time"
    },
    {
      "field": "created_at",
      "format": "date_time"
    },
    {
      "field": "retweeted_status.created_at",
      "format": "date_time"
    },
    {
      "field": "retweeted_status.user.created_at",
      "format": "date_time"
    },
    {
      "field": "user.created_at",
      "format": "date_time"
    }
  ],
  "_source": {
    "excludes": []
  },
  "query": {
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
                        "full_text": "miami"
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
      "should": [],
      "must_not": []
    }
  },
  "highlight": {
    "pre_tags": [
      "@kibana-highlighted-field@"
    ],
    "post_tags": [
      "@/kibana-highlighted-field@"
    ],
    "fields": {
      "*": {}
    },
    "fragment_size": 2147483647
  }
}
```