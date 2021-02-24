# Extract terms about covid-19 in tweets

This project aims to extract terms about covid-19 in the corpus of tweets created by Emily Chen : https://github.com/echen102/COVID-19-TweetIDs

## Pre-requisites :
Tweets have to be download (i.e. hydrated) from Echen repository and indexed into an Elasticsearch index. See steps belows :
1. Git pull echen directory (https://github.com/echen102/COVID-19-TweetIDs)
2. Install twarc from pip and configure with a twitter account. Cf : https://github.com/DocNow/twarc
3. Launch echen hydrate script
4. Copy all hydrating tweets. There are zipped :
        find . -name '*.jsonl.gz' -exec cp -prv '{}' 'hydrating-and-extracting' ';'
5. Unzip all json.gz : `gunzip hydrating-and-extracting/coronavirus-tweet`
6. Index in a Elastic Search  :
    1. Installation of ELK and plugins :
        * Install ELK : logstash, elasticsearch and kibana
        * Install a plugin for logstash to geocode user location (plugin is for using API Rest):
            `sudo /usr/share/logstash/bin/logstash-plugin install logstash-filter-rest`
    2. Start indexing in elastic with logstash :
        * `sudo /usr/share/logstash/bin/logstash -f elasticsearch/logstash-config/json.conf`
        * /!\ Be carefull if you try to index with a laptop using Wifi, it may power off wlan interface even if you desable sleep mode. If you are using a debian/ubuntu OS, you'll need to disable power management on your wlan interface. =>
        `sudo iwconfig wlo1 power off` (non permanent on reboot)
    3. (OPTIONAL) : Kibana : you can import [dashboard](elasticsearch/kibana-dashboard)
    
## Run the main script:
The following script allows to :
+ Build a Hiearchical TF-IDF called H-TFIDF over space and time
+ Build classical TF-IDF to compare with
+ Encode both extracted terms from previous measures to compute semantic similarity :

[COVID-19-TweetIDS-ES-Analyse.py](COVID-19-TweetIDS-ES-Analyse.py)

More experimentations or methods for evaluate H-TFIDF compared with a classical TF-IDF can be found [script]() and [explaination](readme_ressources/eda_es_md)

## OPTIONAL Script:
in order to **explore the dataset without using elastic search** (except from one of them), here are some scripts that allow to have first results :

* [Dataset Analysis](exploration_data_analyse/COVID-19-TweetIDs-dataset-analyse.py) : some stats computed on Echen corpus
* [Extractor](exploration_data_analyse/COVID-19-TweetIDs-extractor.py) : script extracting only tweets'contents (without RT) in order to share data without all twitter's verbose API 
* And a Pipeline for terms extraction using biotex :
    1. [preprocess](exploration_data_analyse/COVID-19-TweetIDs-preprocess.py) : cleaning up tweets and building corpus in the biotex syntaxe
    2. [biotex-wrapper](exploration_data_analyse/COVID-19-TweetsIDS_biotex_wrapper.py): An automatisation of biotex on 4 settings
    3. [merge biotex results](exploration_data_analyse/COVID-19-TweetIDS-merge-biotex-results.py): Due to the size of this corpus, biotex could not be launched on the full corpus. It has to be splitt in 30k tweets. Then results have to be merged and ranked
* [Other fonctions to explore but which use Elasticsearch](exploration_data_analyse/eda-es.py)
    
This is based upon works of:
* **Juan Antonio LOSSIO-VENTURA** creator of [BioTex](https://github.com/sifrproject/biotex/tree/master)
* **Jacques Fize** who build a python wrapper of Biotext (see [his repository](https://gitlab.irstea.fr/jacques.fize/biotex_python) for more details)
* **Gaurav Shrivastava** who code FASTR algorithme in python. His script is in this repository

NB : **Due to the size of this corpus, biotex could not be launched on the full corpus. It has to be splitt in 30k tweets. Then results have to be merged and ranked**
