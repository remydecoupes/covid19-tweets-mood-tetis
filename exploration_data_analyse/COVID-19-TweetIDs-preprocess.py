#!/usr/bin/env python

"""
Hydrate tweets from tweetIDS : https://github.com/echen102/COVID-19-TweetIDs
with twarc configured with Sarah Valentin personnal account : https://github.com/DocNow/twarc

Don't foget to git pull on a terminal :
/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/echen102_COVID19-TweetIDs/COVID-19-TweetIDs && git pull


then launch echen hydrate script

Copie des json.gz dans un autre répertoire : 
    find . -name '*.jsonl.gz' -exec cp -prv '{}' 'hydrating-and-extracting' ';'

Unzip des json.gz :
    gunzip hydrating-and-extracting/coronavirus-tweet-id-2020-01-*

Lancer ce présent script

Lancer biotex depuis la VM :
    copier le corpus vers le répertoire de corpus de biotex
    changer le biotex.properties pour pointer sur le bon corpus
    java -Xms6g -Xmx10g -jar biotex/target/biotex.jar biotex/biotex.properties
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import timedelta, date
import re
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from langdetect import detect
import numpy as np
import time
from random import sample

langs = ['english', 'french', 'spanish', 'other']


def daterange(start_date, end_date):
    """
    Iterator on date range
    :param start_date:
    :param end_date:
    :return:
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


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

    ## 2 - detection langue
    try:
        lang = detect(textclean)
    except:
        lang = "error"
    # Test remove lang detection because of computation time
    # lang = 'en'

    ## 3 - remove stopwords :
    # textclean = stopwords.words('english') + stopwords.words('french') + stopwords.words('spanish')
    return [textclean, lang]


def dailyTweetsToBioTexCorpus(directory, day, biotexcopruspath):
    """
    Build a biotex corpus of a day of tweets
    :param directory: directory where tweets are
    :param day: the day to compute
    :return: None
    """
    print("day: " + day.strftime("%Y-%m-%d"))
    biotexcorpuslang = dict.fromkeys(langs)
    for lang in langs:
        biotexcorpuslang[lang] = ''
    for file in directory.glob('coronavirus-tweet-id-' + day.strftime("%Y-%m-%d") + '*.jsonl'):
        with open(file, 'r') as f:
            print(f)
            for line in f:
                tweet = json.loads(line)
                tweetclean = preprocessTweets(tweet['full_text'])
                if tweetclean[1] == 'en':
                    biotexcorpuslang['english'] += tweetclean[0]
                    biotexcorpuslang['english'] += '\n'
                    biotexcorpuslang['english'] += "##########END##########"
                    biotexcorpuslang['english'] += '\n'
                elif tweetclean[1] == 'fr':
                    biotexcorpuslang['french'] += tweetclean[0]
                    biotexcorpuslang['french'] += '\n'
                    biotexcorpuslang['french'] += "##########END##########"
                    biotexcorpuslang['french'] += '\n'
                elif tweetclean[1] == 'es':
                    biotexcorpuslang['spanish'] += tweetclean[0]
                    biotexcorpuslang['spanish'] += '\n'
                    biotexcorpuslang['spanish'] += "##########END##########"
                    biotexcorpuslang['spanish'] += '\n'
                else:
                    biotexcorpuslang['other'] += tweetclean[0]
                    biotexcorpuslang['other'] += '\n'
                    biotexcorpuslang['other'] += "##########END##########"
                    biotexcorpuslang['other'] += '\n'
    for lang in langs:
        corpusfilename = lang + "-tweetsOf-" + day.strftime("%Y-%m-%d")
        biotexCorpusPath = str(biotexcopruspath) + '/' + corpusfilename
        print(biotexCorpusPath)
        f = open(biotexCorpusPath, 'w')
        f.write(biotexcorpuslang[lang])
        f.close()


def dailyTweetsToBioTexCorpusOptimized(directory, day, biotexcopruspath):
    """
    Build a biotex corpus of a day of tweets :
    Optimized : because string are immutable, we use list
    :param directory: directory where tweets are
    :param day: the day to compute
    :return: None
    """
    print("Compute day: " + day.strftime("%Y-%m-%d"))
    biotexcorpuslang = dict.fromkeys(langs)
    for lang in langs:
        biotexcorpuslang[lang] = []
    for file in directory.glob('coronavirus-tweet-id-' + day.strftime("%Y-%m-%d") + '*.jsonl'):
        with open(file, 'r') as f:
            # print("Read File : "+f.name)
            for line in f:
                tweet = json.loads(line)
                tweetclean = preprocessTweets(tweet['full_text'])
                if tweetclean[1] == 'en':
                    biotexcorpuslang['english'].append(tweetclean[0])
                    biotexcorpuslang['english'].append('\n')
                    biotexcorpuslang['english'].append("##########END##########")
                    biotexcorpuslang['english'].append('\n')
                elif tweetclean[1] == 'fr':
                    biotexcorpuslang['french'].append(tweetclean[0])
                    biotexcorpuslang['french'].append('\n')
                    biotexcorpuslang['french'].append("##########END##########")
                    biotexcorpuslang['french'].append('\n')
                elif tweetclean[1] == 'es':
                    biotexcorpuslang['spanish'].append(tweetclean[0])
                    biotexcorpuslang['spanish'].append('\n')
                    biotexcorpuslang['spanish'].append("##########END##########")
                    biotexcorpuslang['spanish'].append('\n')
                else:
                    biotexcorpuslang['other'].append(tweetclean[0])
                    biotexcorpuslang['other'].append('\n')
                    biotexcorpuslang['other'].append("##########END##########")
                    biotexcorpuslang['other'].append('\n')
    for lang in langs:
        # rebuilds string for save
        textToSave = "".join(biotexcorpuslang[lang])
        corpusfilename = lang + "-tweetsOf-" + day.strftime("%Y-%m-%d")
        biotexCorpusPath = str(biotexcopruspath) + '/' + corpusfilename
        print("\t saving file : " + str(biotexCorpusPath))
        f = open(biotexCorpusPath, 'w')
        f.write(textToSave)
        f.close()


def SubdividedTweetsToBioTexCorpusOptimizedWithoutRT(directory, sizeOfCorpus, startday, endday, biotexcopruspath):
    """
    Build multiple biotex corpus of sizeOfCorpus :
    Optimized : because string are immutable, we use list
    WithoutRT : because RT are removed
    :param directory: path to tweets
    :param sizeOfCorpus: nb of tweets of subdivied corpus
    :param startday: Period starts
    :param endday: Period ends
    :param biotexcopruspath: pathToSave
    :return:
    """
    biotexcorpuslang = dict.fromkeys(langs)
    nbOFtweets = dict.fromkeys(langs)
    nbOfCorpus = dict.fromkeys(langs)
    retweet_pattern = 'retweeted_status'
    for lang in langs:
        biotexcorpuslang[lang] = []
        nbOFtweets[lang] = 0
        nbOfCorpus[lang] = 0
    for single_date in daterange(startday, endday):
        print("Compute day: " + single_date.strftime("%Y-%m-%d"))
        for file in directory.glob('coronavirus-tweet-id-' + single_date.strftime("%Y-%m-%d") + '*.jsonl'):
            with open(file, 'r') as f:
                # print("Read File : "+f.name)
                for line in f:
                    tweet = json.loads(line)
                    if retweet_pattern not in tweet:  # it's not a RT
                        tweetclean = preprocessTweets(tweet['full_text'])
                        if tweetclean[1] == 'en':
                            biotexcorpuslang['english'].append(tweetclean[0])
                            biotexcorpuslang['english'].append('\n')
                            biotexcorpuslang['english'].append("##########END##########")
                            biotexcorpuslang['english'].append('\n')
                            nbOFtweets['english'] += 1
                        elif tweetclean[1] == 'fr':
                            biotexcorpuslang['french'].append(tweetclean[0])
                            biotexcorpuslang['french'].append('\n')
                            biotexcorpuslang['french'].append("##########END##########")
                            biotexcorpuslang['french'].append('\n')
                            nbOFtweets['french'] += 1
                        elif tweetclean[1] == 'es':
                            biotexcorpuslang['spanish'].append(tweetclean[0])
                            biotexcorpuslang['spanish'].append('\n')
                            biotexcorpuslang['spanish'].append("##########END##########")
                            biotexcorpuslang['spanish'].append('\n')
                            nbOFtweets['spanish'] += 1
                        else:
                            biotexcorpuslang['other'].append(tweetclean[0])
                            biotexcorpuslang['other'].append('\n')
                            biotexcorpuslang['other'].append("##########END##########")
                            biotexcorpuslang['other'].append('\n')
                            nbOFtweets['other'] += 1

                        # Save corpus if nb of tweets = sizeOfCorpus
                        for lang in langs:
                            # rebuilds string for save
                            if nbOFtweets[lang] == sizeOfCorpus:
                                nbOfCorpus[lang] += 1
                                textToSave = "".join(biotexcorpuslang[lang])
                                corpusfilename = lang + "-subdividedcorpus-" + str(nbOfCorpus[lang])
                                biotexCorpusPath = str(biotexcopruspath) + '/' + corpusfilename
                                print("\t saving file : " + str(biotexCorpusPath))
                                f = open(biotexCorpusPath, 'w')
                                f.write(textToSave)
                                f.close()
                                nbOFtweets[lang] = 0
                                biotexcorpuslang[lang] = []
        # We do not save few tweets if size of subdivided corpus < sizeOfCorpus


def pythonTFIDF(directory, outputdir):
    """
    :param directory:
    -----
    Be aware :
    TF.IDF can't be compute using hash method. So working on tweets, matrice sizes are to big.
    Exemple on 4 Millions tweets : python error : MemoryError: Unable to allocate 41.8 TiB for an array with shape
    (4237779, 1356026) and data type float64
    -----
    :return:
    """
    listOfDocLang = dict.fromkeys(langs)
    for lang in langs:
        listOfDocLang[lang] = []

    for file in directory.glob('coronavirus-tweet-id-2020-01-25*.jsonl'):
        with open(file, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                tweetclean = preprocessTweets(tweet['full_text'])
                if tweetclean[1] == 'en':
                    # listOfDocLang['english'].append(remove_stopwords(tweetclean[0]))
                    listOfDocLang['english'].append(tweetclean[0])
                elif tweetclean[1] == 'fr':
                    listOfDocLang['french'].append(tweetclean[0])
                elif tweetclean[1] == 'es':
                    listOfDocLang['spanish'].append(tweetclean[0])
                else:
                    listOfDocLang['other'].append(tweetclean[0])

    for lang in langs:
        vectorizer = TfidfVectorizer()
        if listOfDocLang[lang] is None or lang == 'other':  # list vide or langue not found
            print("No vocabulary for :" + lang)
        else:
            debut = time.time()
            vectors = vectorizer.fit_transform(listOfDocLang[lang])
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
            print("\t sklearn/TF-IDF : Done in {0} sec".format(time.time() - debut))
            df = pd.DataFrame(denselist, columns=feature_names)
            df.to_csv(str(outputdir) + '/tfidf_' + lang + '.csv')
            # print top 20 TF.IDF :
            if len(df.keys()) < 20:
                print("TFIDF score for " + lang)
                # print(df.sort_values(by=1, ascending=False, axis=1).keys())
                # print(vectors)
            else:
                print("TOP 20 higher score TDIDF for " + lang)
                # print(df.sort_values(by=1, ascending=False, axis=1).keys()[0:20])
                # print(vectors[0:20])
    return 1


def randomSampling4PythonTFIDF(directory, outputdir):
    """
    :param directory:
    -----
    Be aware :
    TF.IDF can't be compute using hash method. So working on tweets, matrice sizes are to big.
    Exemple on 4 Millions tweets : python error : MemoryError: Unable to allocate 41.8 TiB for an array with shape
    (4237779, 1356026) and data type float64
    -----
    :return:
    """
    listOfDocLang = dict.fromkeys(langs)
    for lang in langs:
        listOfDocLang[lang] = []

    for file in directory.glob('coronavirus-tweet-id-2020-01-25*.jsonl'):
        with open(file, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                tweetclean = preprocessTweets(tweet['full_text'])
                if tweetclean[1] == 'en':
                    # listOfDocLang['english'].append(remove_stopwords(tweetclean[0]))
                    listOfDocLang['english'].append(tweetclean[0])
                elif tweetclean[1] == 'fr':
                    listOfDocLang['french'].append(tweetclean[0])
                elif tweetclean[1] == 'es':
                    listOfDocLang['spanish'].append(tweetclean[0])
                else:
                    listOfDocLang['other'].append(tweetclean[0])

    for lang in langs:
        vectorizer = TfidfVectorizer()
        if listOfDocLang[lang] is None or lang == 'other' or len(
                listOfDocLang[lang]) < 30000:  # list vide or langue not found
            print("No vocabulary for :" + lang)
        else:
            listOfDocLang[lang] = sample(listOfDocLang[lang], 50000)
            debut = time.time()
            vectors = vectorizer.fit_transform(listOfDocLang[lang])
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
            print("\t sklearn/TF-IDF : Done in {0} sec".format(time.time() - debut))
            df = pd.DataFrame(denselist, columns=feature_names)
            df.to_csv(str(outputdir) + '/randomSampling_tfidf_' + lang + '.csv')
            # print top 20 TF.IDF :
            if len(df.keys()) < 20:
                print("TFIDF score for " + lang)
                # print(df.sort_values(by=1, ascending=False, axis=1).keys())
                # print(vectors)
            else:
                print("TOP 20 higher score TDIDF for " + lang)
                # print(df.sort_values(by=1, ascending=False, axis=1).keys()[0:20])
                # print(vectors[0:20])
    return 1


if __name__ == '__main__':
    hydrateTweetsDir = Path('../hydrating-and-extracting')
    biotexcopruspath = Path('../biotexcorpus')
    tdidfpythonpath = Path('../tfidfpython')

    print("Begin")
    ## Test biotex on 4M tweets : doesn't work
    # def get_full_text_from_tweets(directory):
    #     """
    #
    #     :param directory: path to directory where json twarc unzipped
    #     :return: biotexcorpus : concate all full text from tweets with biotex separator
    #     """
    #     biotexcorpus = ''
    #     for file in directory.glob('*.jsonl'):
    #         with open(file, 'r') as f:
    #             for line in f:
    #                 tweet = json.loads(line)
    #                 biotexcorpus += tweet['full_text']
    #                 biotexcorpus += '\n'
    #                 biotexcorpus += "##########END##########"
    #                 biotexcorpus += '\n'
    #
    #     return biotexcorpus
    # biotexCorpusPath = 'biotexcorpus/biotexcorpus.txt'
    # biotexcorpus = get_full_text_from_tweets(hydrateTweetsDir)
    # f = open(biotexCorpusPath, 'w')
    # f.write(biotexcorpus)
    # f.close()
    ## end of Test biotex on 4M tweets

    ## Test TF.IDF from sklearn on 4M tweets : doen't work : to big : need 40 TB...
    # df = pythonTFIDF(hydrateTweetsDir, tdidfpythonpath)

    ## Test TF.IDF from sklearn on random tweets per day :
    # df = randomSampling4PythonTFIDF(hydrateTweetsDir, tdidfpythonpath)

    ## Daily tweets
    # startDate = date(2020, 1, 30)
    # enDate = date(2020, 2, 1)
    # for single_date in daterange(startDate,enDate):
    #     dailyTweetsToBioTexCorpusOptimized(hydrateTweetsDir, single_date, biotexcopruspath)

    ## Subdivided corpus of Size = ??
    startDate = date(2020, 1, 22)
    enDate = date(2020, 2, 12)
    sizeOfCorpus = 30000
    biotexcopruspath = Path('../biotexcorpus/subdividedcorpus')
    SubdividedTweetsToBioTexCorpusOptimizedWithoutRT(hydrateTweetsDir, sizeOfCorpus, startDate, enDate,
                                                     biotexcopruspath)

    print("end")
