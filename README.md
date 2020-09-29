# Extract terms about covid-19 in tweets

This project aims to extracts terms about covid-19 in the corpus of tweets created by Echen : https://github.com/echen102/COVID-19-TweetIDs

This repository is based upon works of:
* **Juan Antonio LOSSIO-VENTURA** creator of [BioTex](https://github.com/sifrproject/biotex/tree/master)
* **Jacques Fize** who build a python wrapper of Biotext (see [his repository](https://gitlab.irstea.fr/jacques.fize/biotex_python) for more details)
* **Gaurav Shrivastava** who code FASTR algorithme in python. His script is in this repository

NB : **Due to the size of this corpus, biotex could not be launched on the full corpus. It has to be splitt in 30k tweets. Then results have to be merged and ranked**

## Pre-requisites :
Tweets have to be download (i.e. hydrated) from Echen repository. See steps belows :
1. Git pull echen directoru (https://github.com/echen102/COVID-19-TweetIDs)
2. Install twarc from pip and configure with a twitter account. Cf : https://github.com/DocNow/twarc
3. Launch echen hydrate script
4. Copy all hydrating tweets. There are zipped :
        find . -name '*.jsonl.gz' -exec cp -prv '{}' 'hydrating-and-extracting' ';'
5. Unzip all json.gz :
        gunzip hydrating-and-extracting/coronavirus-tweet
        
   
## Description of python scripts
* [Dataset Analysis](COVID-19-TweetIDs-dataset-analyse.py) : some stats computed on Echen corpus
* [Extractor](COVID-19-TweetIDs-extractor.py) : script extracting only tweets'contents (without RT) in order to share data without all twitter's verbose API 
* And a Pipeline for terms extraction using biotex :
    1. [preprocess](COVID-19-TweetIDs-preprocess.py) : cleaning up tweets and building corpus in the biotex syntaxe
    2. [biotex-wrapper](COVID-19-TweetsIDS_biotex_wrapper.py): An automatisation of biotex on 4 settings
    3. [merge biotex results](COVID-19-TweetIDS-merge-biotex-results.py): Due to the size of this corpus, biotex could not be launched on the full corpus. It has to be splitt in 30k tweets. Then results have to be merged and ranked
    
## Indexing and using Elasticsearch to explore corpus twitter API metadata
1. Installation of ELK and plugins :
    * Install ELK : logstash, elasticsearch and kibana
    * Install a plugin for logstash to geocode user location (plugin is for using API Rest):
        sudo /usr/share/logstash/bin/logstash-plugin install logstash-filter-rest
2. Start indexing in elastic with logstash :
    * sudo /usr/share/logstash/bin/logstash -f elasticsearch/logstash-config/json.conf
3. Import into kibana : dashbord from this repository
4. [Query](elasticsearch/analyse/Elasticquery.md) elasticsearch to extract tweets inside a specific spatio-temporal windows
5. [Build an adaptive TF-IDF on this extraction of tweets & analyse results](COVID-19-TweetIDS-ES-Analyse.py)
6. Match term extracted by our various methods with thesaurus like Agrovoc or MeSH using [SPARQL](elasticsearch/analyse/sparqlquery.md)
7. For the 3 measures first 100 terms : compares occurence of terms in states aggregated tweets
8. For the K most frequent terms in state aggregated tweets : compute % of the 3 measures coverage
9. Evaluation with different TF / TF-IDF settings : work on corpus and document 

| H-TF-IDF  |  TF / TF-IDF UK | TF / TF-IDF by state |
|:---|:---|:---|
| Doc: All tweets aggregate by state  | Doc : 1 tweet  | Doc : 1 tweet  |
| Corpus: UK  | Corpus: UK  | Corpus: for each state  |
