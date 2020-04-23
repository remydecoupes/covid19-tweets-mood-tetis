#!/usr/bin/env python

"""
Steps for extracted full_text from tweets :
    1 - Git pull echen directoru (https://github.com/echen102/COVID-19-TweetIDs)
    2 - Install twarc from pip and configure with a twitter account. Cf : https://github.com/DocNow/twarc
    3 - Launch echen hydrate script
    4 - Copy all hydrating tweets. There are zipped :
            find . -name '*.jsonl.gz' -exec cp -prv '{}' 'hydrating-and-extracting' ';'
    5 - Unzip all json.gz :
            gunzip hydrating-and-extracting/coronavirus-tweet
    6 - Then launch this present script
"""
import json
from pathlib import Path
from collections import defaultdict

def get_full_text_from_tweets(directory):
    """
    Extract all tweets' text from hydrated echen tweets
    :param directory: path to directory where json from twarc are unzipped
    :return: string : concatenate all full text from tweets with biotex separator
    """
    retweet_pattern = 'retweeted_status'
    tweetFullText = []
    for file in directory.glob('*.jsonl'):
        with open(file, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if retweet_pattern not in tweet:  # it's not a RT
                    tweetFullText.append(tweet['full_text'])
                    tweetFullText.append('\n')
                    tweetFullText.append("##########END##########")
                    tweetFullText.append('\n')
    strTweetFullText = "".join(tweetFullText)
    return strTweetFullText

def extract_json_from_tweets(directory):
    """
    Extract all tweets' text from hydrated echen tweets
    :param directory: path to directory where json from twarc are unzipped
    :return: string : concatenate all full text from tweets with biotex separator
    """
    retweet_pattern = 'retweeted_status'
    tweetExtract = {
        "type": "FeatureCollection",
        "features": []
    }
    for file in directory.glob('*.jsonl'):
        with open(file, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if retweet_pattern not in tweet:  # it's not a RT
                    json_feature = {
                        "tweetID": tweet['id'],
                        "full_text": tweet['full_text']
                    }
                    tweetExtract['features'].append(json_feature)
    return tweetExtract



if __name__ == '__main__':
    hydrateTweetsDir = Path('hydrating-and-extracting')
    extractedTweetsDir = Path('extractedTweetsWithoutPreprocess')

    print("Begin tweets extraction")
    # # Extract full_text
    # tweetFullText = get_full_text_from_tweets(hydrateTweetsDir)
    # filename = str(extractedTweetsDir) + "/extractFullTweetsWithoutRT.txt"
    # print("\t saving file : " + str(filename))
    # f = open(filename, 'w')
    # f.write(tweetFullText)
    # f.close()

    # build a json with tweets id and full_text
    jsondict = extract_json_from_tweets(hydrateTweetsDir)
    filename = str(extractedTweetsDir) + "/extractTweetsWithoutRT.json"
    # Serializing json
    json_object = json.dumps(jsondict, indent=4)
    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)
    print("End of tweets extraction")
