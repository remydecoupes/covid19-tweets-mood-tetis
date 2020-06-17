#!/usr/bin/env python

"""
Stat on dataset
"""
from datetime import timedelta, date
from pathlib import Path
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import json
import geopandas as gpd
import time

def daterange(start_date, end_date):
    """
    Iterator on date range
    :param start_date:
    :param end_date:
    :return:
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def nbOfTweetsPerDay(day, tweetsIdFiles):
    """

    :param day: processing day (date)
    :param tweetsIdFiles: files for echen102
    :return:
    """
    num_lines_day = 0
    for root, dirs, files in os.walk(tweetsIdFiles):
        for file in files:
            if file.startswith('coronavirus-tweet-id-'+day.strftime("%Y-%m-%d")) and file.endswith('.txt'):
                filepath = os.path.join(root, file)
                num_lines = sum(1 for line in open(filepath))
                num_lines_day += num_lines
    return num_lines_day

def plotTweetsFromEchen():
    """

    :return:
    """
    listOfTweetsPerDay = defaultdict()
    startDate = date(2020, 1, 22)
    enDate = date(2020, 2, 11)
    tweetsDir = Path('echen102_COVID19-TweetIDs/COVID-19-TweetIDs')
    for single_date in daterange(startDate, enDate):
        listOfTweetsPerDay[single_date.strftime("%Y-%m-%d")] = nbOfTweetsPerDay(single_date, tweetsDir)
        #listOfTweetsPerDay[single_date] = nbOfTweetsPerDay(single_date, tweetsDir)
    #print(listOfTweetsPerDay)
    # Plot
    plt.style.use('ggplot')
    series = pd.Series(listOfTweetsPerDay)
    series.to_csv("dataset-analyse.csv")
    #series.index = pd.DateTimeIndex(series.index)

    fig, ax = plt.subplots(figsize=(15, 7))
    series.plot(kind='bar', x='date', y='number of tweets per day', ax=ax,
                title='Number of tweets from Echen102/COVID-19-TweetIDs per day on period 3')
    plt.savefig('numberOfTweetsPerDayPeriod3.png')
    plt.show()


def nbOfTweetsAndRTPerDay(day, hydrateDir):
    """

    :param day: processing day (date)
    :param tweetsIdFiles: files for echen102
    :return:
    """
    num_tweets = 0
    num_rt = 0
    retweet_pattern = 'retweeted_status'
    for root, dirs, files in os.walk(hydrateDir):
        for file in files:
            if file.startswith('coronavirus-tweet-id-'+day.strftime("%Y-%m-%d")) and file.endswith('.jsonl'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    for line in f:
                        if retweet_pattern in line:
                            num_rt += 1
                        else:
                            num_tweets +=1
    return [day, num_tweets, num_rt]

def plotTweetsAndRT():
    """

    :return:
    """
    listOfTweetsPerDay = defaultdict()
    startDate = date(2020, 1, 22)
    enDate = date(2020, 2, 13)
    tweetsHydrateDir = Path('hydrating-and-extracting')
    df = pd.DataFrame(columns=('date', 'tweets', 'retweets'))
    i = 0
    for single_date in daterange(startDate, enDate):
        df.loc[i] = nbOfTweetsAndRTPerDay(single_date, tweetsHydrateDir)
        i += 1

    ## Plot
    plt.style.use('ggplot')
    df.to_csv("dataset-analyse.csv")

    fig, ax = plt.subplots(figsize=(15, 7))
    # ax = df.plot(kind='bar', x='date', y='tweets', color="C2",
    #              title='Number of tweets and retweets from Echen102/COVID-19-TweetIDs per day on period 3')
    # df.plot(kind='bar', x='date', y='retweets', ax=ax)
    figure_title = 'Number of tweets and retweets from Echen102/COVID-19-TweetIDs per day on period 3'
    axsub = df.plot(kind='bar',
            x='date',
            y=['tweets', 'retweets'],
            title=figure_title,
            ax=ax
    )
    ### Change x and y label
    axsub.set_xlabel("Date")
    axsub.set_ylabel("Number of tweets or retweets")

    ### Raised title otherwise it overlaps on legend
    plt.title(figure_title, y=1.08)
    plt.savefig('numberOfTweetsPerDayPeriod3.png')
    plt.show()

def tweetLocationExtract(geoJsonfile):
    """
    Adapt from https://marcobonzanini.com/2015/06/16/mining-twitter-data-with-python-and-js-part-7-geolocation-and-interactive-maps/
    :return:
    """
    # startDate = date(2020, 1, 22)
    # enDate = date(2020, 2, 11)
    tweetsHydrateDir = Path('hydrating-and-extracting')
    geo_data = {
        "type": "FeatureCollection",
        "features": []
    }
    for root, dirs, files in os.walk(tweetsHydrateDir):
        for file in files:
            if file.startswith('coronavirus-tweet-id-') and file.endswith('.jsonl'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    for line in f:
                        tweet = json.loads(line)
                        if tweet['coordinates']:
                            try: # is-it a RT ?
                                test = tweet['retweeted_status']
                                geo_json_feature = {
                                    "type": "Feature",
                                    "geometry": tweet['coordinates'],
                                    "properties": {
                                        #"text": tweet['full_text'],
                                        "tweetID": tweet['id'],
                                        "rt": True,
                                        "quoted": tweet['is_quote_status'],
                                        # parse Date into iso8601
                                        "created_at": time.strftime('%Y-%m-%d %H:%M:%S',
                                                                    time.strptime(tweet['created_at'],
                                                                                  '%a %b %d %H:%M:%S +0000 %Y'))
                                    }
                                }
                            except: #Is not a RT
                                geo_json_feature = {
                                    "type": "Feature",
                                    "geometry": tweet['coordinates'],
                                    "properties": {
                                        #"text": tweet['full_text'],
                                        "tweetID": tweet['id'],
                                        "rt": False,
                                        "quoted": tweet['is_quote_status'],
                                        # parse Date into iso8601
                                        "created_at": time.strftime('%Y-%m-%d %H:%M:%S',
                                                                    time.strptime(tweet['created_at'],
                                                                                  '%a %b %d %H:%M:%S +0000 %Y'))
                                    }
                                }
                            geo_data['features'].append(geo_json_feature)
        print(geo_data)
        with open(geoJsonfile, 'w') as fout:
            fout.write(json.dumps(geo_data, indent=4))


if __name__ == '__main__':
    print("begin")
    listOfTweetsPerDay = defaultdict()
    startDate = date(2020, 1, 22)
    enDate = date(2020, 2, 13)

    ## Count nb of tweets from github echen
    # plotTweetsFromEchen()

    ## Count nb of tweets and retweet from hydrating and extracting
    plotTweetsAndRT()

    ## Stat on location
    geoJsonfile = 'COVID19_echen_geo_data.json'
    tweetLocationExtract(geoJsonfile)
    geodata = gpd.read_file(geoJsonfile)
    geodata.plot()


    print("end")
