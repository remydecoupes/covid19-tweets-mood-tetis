# List of usefull queries
## Get all tweets from United Kingdom 
### Kibana dev tools
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
### cURL syntax :
curl -XGET "http://localhost:9200/twitter/_search?scroll=1m&pretty" -H 'Content-Type: application/json' -d'{  "size": 10000,  "query": {    "bool": {      "must": [        {          "match" :{            "rest_user_osm.country.keyword" : "United Kingdom"           }        },        {          "range": {            "created_at": {              "gte": "Wed Jan 22 00:00:01 +0000 2020"            }          }        }]    }  }}'