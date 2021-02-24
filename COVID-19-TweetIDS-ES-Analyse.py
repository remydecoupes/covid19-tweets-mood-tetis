#!/usr/bin/env python

"""
analyse Elasticsearch query
"""
import json
from elasticsearch import Elasticsearch
from elasticsearch import logger as es_logger
from collections import defaultdict, Counter
import re
import os
from datetime import datetime
# Preprocess terms for TF-IDF
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# progress bar
from tqdm import tqdm
# ploting
import matplotlib.pyplot as plt
# LOG
import logging
from logging.handlers import RotatingFileHandler
# Word embedding for evaluation
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import scipy.spatial as sp
# Spatial entity as descriptor :
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Global var on Levels on spatial and temporal axis
spatialLevels = ['city', 'state', 'country']
temporalLevels = ['day', 'week', 'month', 'period']


def elasticsearch_query(query_fname, logger):
    """
    Build a ES query  and return a default dict with resuls
    :return: tweetsByCityAndDate
    """
    # Elastic search credentials
    client = Elasticsearch("http://localhost:9200")
    es_logger.setLevel(logging.WARNING)
    index = "twitter"
    # Define a Query
    query = open(query_fname, "r").read()
    result = Elasticsearch.search(client, index=index, body=query, scroll='2m', size=5000)

    # Append all pages form scroll search : avoid the 10k limitation of ElasticSearch
    results = avoid10kquerylimitation(result, client, logger)

    # Initiate a dict for each city append all Tweets content
    tweetsByCityAndDate = defaultdict(list)
    for hits in results:
        # parse Java date : EEE MMM dd HH:mm:ss Z yyyy
        inDate = hits["_source"]["created_at"]
        parseDate = datetime.strptime(inDate, "%a %b %d %H:%M:%S %z %Y")

        try:# geodocing may be bad
            geocoding = hits["_source"]["rest"]["features"][0]["properties"]
        except:
            continue # skip this iteraction
        if "country" in hits["_source"]["rest"]["features"][0]["properties"]:
            # locaties do not necessarily have an associated stated
            try:
                cityStateCountry = str(hits["_source"]["rest"]["features"][0]["properties"]["city"]) + "_" + \
                                   str(hits["_source"]["rest"]["features"][0]["properties"]["state"]) + "_" + \
                                   str(hits["_source"]["rest"]["features"][0]["properties"]["country"])
            except: # there is no state in geocoding
                try:
                    logger.debug(hits["_source"]["rest"]["features"][0]["properties"]["city"] + " has no state")
                    cityStateCountry = str(hits["_source"]["rest"]["features"][0]["properties"]["city"]) + "_" + \
                                       str("none") + "_" + \
                                       str(hits["_source"]["rest"]["features"][0]["properties"]["country"])
                except: # there is no city as well : only country
                    # print(json.dumps(hits["_source"], indent=4))
                    try:  #
                        cityStateCountry = str("none") + "_" + \
                                           str("none") + "_" + \
                                           str(hits["_source"]["rest"]["features"][0]["properties"]["country"])
                    except:
                        cityStateCountry = str("none") + "_" + \
                                           str("none") + "_" + \
                                           str("none")
        try:
            tweetsByCityAndDate[cityStateCountry].append(
                {
                    "tweet": preprocessTweets(hits["_source"]["full_text"]),
                    "created_at": parseDate
                }
            )
        except:
            print(json.dumps(hits["_source"], indent=4))
    # biotexInputBuilder(tweetsByCityAndDate)
    # pprint(tweetsByCityAndDate)
    return tweetsByCityAndDate


def avoid10kquerylimitation(result, client, logger):
    """
    Elasticsearch limit results of query at 10 000. To avoid this limit, we need to paginate results and scroll
    This method append all pages form scroll search
    :param result: a result of a ElasticSearcg query
    :return:
    """
    scroll_size = result['hits']['total']["value"]
    logger.info("Number of elasticsearch scroll: " + str(scroll_size))
    results = []
    # Progress bar
    pbar = tqdm(total=scroll_size)
    while (scroll_size > 0):
        try:
            scroll_id = result['_scroll_id']
            res = client.scroll(scroll_id=scroll_id, scroll='60s')
            results += res['hits']['hits']
            scroll_size = len(res['hits']['hits'])
            pbar.update(scroll_size)
        except:
            pbar.close()
            logger.error("elasticsearch search scroll failed")
            break
    pbar.close()
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
    # textclean = re.sub('@[^\s]+', '', textclean)
    # remove the # in #hashtag
    # textclean = re.sub(r'#([^\s]+)', r'\1', textclean)
    return textclean

def matrixOccurenceBuilder(tweetsofcity, matrixAggDay_fout, matrixOccurence_fout, logger):
    """
    Create a matrix of :
        - line : (city,day)
        - column : terms
        - value of cells : TF (term frequency)
    Help found here :
    http://www.xavierdupre.fr/app/papierstat/helpsphinx/notebooks/artificiel_tokenize_features.html
    https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    :param tweetsofcity:
    :param matrixAggDay_fout: file to save
    :param matrixOccurence_fout: file to save
    :return:
    """
    # initiate matrix of tweets aggregate by day
    # col = ['city', 'day', 'tweetsList', 'bow']
    col = ['city', 'day', 'tweetsList']
    matrixAggDay = pd.DataFrame(columns=col)
    cityDayList = []

    logger.info("start full_text concatenation for city & day")
    pbar = tqdm(total=len(tweetsofcity))
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
            document = '. \n'.join(matrix.loc[matrix['created_at'].dt.date == day]['tweet'].tolist())
            # Bag of Words and preprocces
            # preproccesFullText = preprocessTerms(document)
            tweetsOfDayAndCity = {
                'city': city,
                'day': day,
                'tweetsList': document
            }
            cityDayList.append(city + "_" + str(day))
            try:
                matrixAggDay = matrixAggDay.append(tweetsOfDayAndCity, ignore_index=True)
            except:
                print("full_text empty after pre-process: "+document)
                continue
        pbar.update(1)
    pbar.close()
    logger.info("Saving file: matrix of full_text concatenated by day & city: "+str(matrixAggDay_fout))
    matrixAggDay.to_csv(matrixAggDay_fout)

    # Count terms with sci-kit learn
    cd = CountVectorizer(
        stop_words='english',
        #preprocessor=sklearn_vectorizer_no_number_preprocessor,
        #min_df=2, # token at least present in 2 cities : reduce size of matrix
        max_features=50000,
        ngram_range=(1, 3),
        token_pattern='[a-zA-Z0-9#@]+', #remove user name, i.e term starting with @ for personnal data issue
        # strip_accents= "ascii" # remove token with special character (trying to keep only english word)
    )
    cd.fit(matrixAggDay['tweetsList'])
    res = cd.transform(matrixAggDay["tweetsList"])
    countTerms = res.todense()
    # create matrix
    ## get terms :
    # voc = cd.vocabulary_
    # listOfTerms = {term for term, index in sorted(voc.items(), key=lambda item: item[1])}
    listOfTerms = cd.get_feature_names()
    ##initiate matrix with count for each terms
    matrixOccurence = pd.DataFrame(data=countTerms[0:, 0:], index=cityDayList, columns=listOfTerms)
    # save to file
    logger.info("Saving file: occurence of term: "+str(matrixOccurence_fout))
    matrixOccurence.to_csv(matrixOccurence_fout)
    return matrixOccurence


def spatiotemporelFilter(matrix, listOfcities='all', spatialLevel='city', period='all', temporalLevel='day'):
    """
    Filter matrix with list of cities and a period

    :param matrix:
    :param listOfcities:
    :param spatialLevel:
    :param period:
    :param temporalLevel:
    :return: matrix filtred
    """
    if spatialLevel not in spatialLevels or temporalLevel not in temporalLevels:
        print("wrong level, please double check")
        return 1
    # Extract cities and period
    ## cities
    if listOfcities != 'all':  ### we need to filter
        ### Initiate a numpy array of False
        filter = np.zeros((1, len(matrix.index)), dtype=bool)[0]
        for city in listOfcities:
            ### edit filter if index contains the city (for each city of the list)
            filter += matrix.index.str.startswith(str(city) + "_")
        matrix = matrix.loc[filter]
    ## period
    if str(period) != 'all':  ### we need a filter on date
        datefilter = np.zeros((1, len(matrix.index)), dtype=bool)[0]
        for date in period:
            datefilter += matrix.index.str.contains(date.strftime('%Y-%m-%d'))
        matrix = matrix.loc[datefilter]
    return matrix


def HTFIDF(matrixOcc, matrixHTFIDF_fname, biggestHTFIDFscore_fname, listOfcities='all', spatialLevel='city',
           period='all', temporalLevel='day'):
    """
    Aggregate on spatial and temporel and then compute TF-IDF

    :param matrixOcc: Matrix with TF already compute
    :param listOfcities: filter on this cities
    :param spatialLevel: city / state / country / world
    :param period: Filter on this period
    :param temporalLevel: day / week (month have to be implemented)
    :return:
    """
    matrixOcc = spatiotemporelFilter(matrix=matrixOcc, listOfcities=listOfcities,
                                     spatialLevel='state', period=period)

    # Aggregate by level
    ## Create 4 new columns : city, State, Country and date
    def splitindex(row):
        return row.split("_")

    matrixOcc["city"], matrixOcc["state"], matrixOcc["country"], matrixOcc["date"] = \
        zip(*matrixOcc.index.map(splitindex))

    if temporalLevel == 'day':
        ## In space
        if spatialLevel == 'city':
            # do nothing
            pass
        elif spatialLevel == 'state' and temporalLevel == 'day':
            matrixOcc = matrixOcc.groupby("state").sum()
        elif spatialLevel == 'country' and temporalLevel == 'day':
            matrixOcc = matrixOcc.groupby("country").sum()
    elif temporalLevel == "week":
        matrixOcc.date = pd.to_datetime((matrixOcc.date)) - pd.to_timedelta(7, unit='d')# convert date into datetime
        ## in space and time
        if spatialLevel == 'country':
            matrixOcc = matrixOcc.groupby(["country", pd.Grouper(key="date", freq="W")]).sum()
        elif spatialLevel == 'state':
            matrixOcc = matrixOcc.groupby(["state", pd.Grouper(key="date", freq="W")]).sum()
        elif spatialLevel == 'city':
            matrixOcc = matrixOcc.groupby(["city", pd.Grouper(key="date", freq="W")]).sum()


    # Compute TF-IDF
    ## compute TF : for each doc, devide count by Sum of all count
    ### Sum fo all count by row
    matrixOcc['sumCount'] = matrixOcc.sum(axis=1)
    ### Devide each cell by these sums
    listOfTerms = matrixOcc.keys()
    matrixOcc = matrixOcc.loc[:, listOfTerms].div(matrixOcc['sumCount'], axis=0)
    ## Compute IDF : create a vector of length = nb of termes with IDF value
    idf = pd.Series(index=matrixOcc.keys(), dtype=float)
    ### N : nb of doucments <=> nb of rows :
    N = matrixOcc.shape[0]
    ### DFt : nb of document that contains the term
    DFt = matrixOcc.astype(bool).sum(axis=0)  # Tip : convert all value in boolean. float O,O will be False, other True
    #### Not a Number when value 0 because otherwise log is infinite
    DFt.replace(0, np.nan, inplace=True)
    ### compute log(N/DFt)
    idf = np.log(N / DFt)
    ## compute TF-IDF
    matrixTFIDF = matrixOcc * idf
    ## remove terms if for all documents value are Nan
    matrixTFIDF.dropna(axis=1, how='all', inplace=True)

    # Save file
    matrixTFIDF.to_csv(matrixHTFIDF_fname)

    # Export N biggest TF-IDF score:
    top_n = 500
    extractBiggest = pd.DataFrame(index=matrixTFIDF.index, columns=range(0, top_n))
    for row in matrixTFIDF.index:
        try:
            row_without_zero =  matrixTFIDF.loc[row]# we remove term with a score = 0
            row_without_zero = row_without_zero[ row_without_zero !=0 ]
            try:
                extractBiggest.loc[row] = row_without_zero.nlargest(top_n).keys()
            except:
                extractBiggest.loc[row] = row_without_zero.nlargest(len(row_without_zero)).keys()
        except:
            logger.debug("H-TFIDF: city "+str(matrixTFIDF.loc[row].name)+ "not enough terms")
    extractBiggest.to_csv(biggestHTFIDFscore_fname+".old.csv")
    # Transpose this table in order to share the same structure with TF-IDF classifical biggest score :
    hbt = pd.DataFrame()
    extractBiggest = extractBiggest.reset_index()
    for index, row in extractBiggest.iterrows():
        hbtrow = pd.DataFrame(row.drop([spatialLevel, "date"]).values, columns=["terms"])
        hbtrow[spatialLevel] = row[spatialLevel]
        hbtrow["date"] = row["date"]
        hbt = hbt.append(hbtrow, ignore_index=True)
    hbt.to_csv(biggestHTFIDFscore_fname)

def TFIDF_TF_with_corpus_state(elastic_query_fname, logger, nb_biggest_terms=500, path_for_filesaved="./",
                               spatial_hiearchy="country", temporal_period='all', listOfCities='all'):
    """
    Compute TFIDF and TF from an elastic query file
    1 doc = 1 tweet
    Corpus = by hiearchy level, i.e. : state or country

    :param elastic_query_fname: filename and path of the elastic query
    :param logger: logger of the main program
    :param nb_biggest_terms: How many biggest term are to keep
    :param spatial_hiearchy: define the size of the corpus : state or country
    :param temporal_period:
    :param listOfCities: If you want to filter out some cities, you can
    :return:
    """
    # tfidfStartDate = date(2020, 1, 23)
    # tfidfEndDate = date(2020, 1, 30)
    # temporal_period = pd.date_range(tfidfStartDate, tfidfEndDate)
    # listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']
    # listOfState = ["England", "Scotland", "Northern Ireland", "Wales"]

    # Query Elasticsearch to get all tweets from UK
    tweets = elasticsearch_query(elastic_query_fname, logger)
    if listOfCities == 'all':
        listOfCities = []
        listOfStates = []
        listOfCountry = []
        for triple in tweetsByCityAndDate:
            splitted = triple.split("_")
            listOfCities.append(splitted[0])
            listOfStates.append(splitted[1])
            listOfCountry.append(splitted[2])
        listOfCities = list(set(listOfCities))
        listOfStates = list(set(listOfStates))
        listOfCountry = list(set(listOfCountry))
    # reorganie tweets (dict : tweets by cities) into dataframe (city and date)
    matrixAllTweets = pd.DataFrame()
    for tweetByCity in tweets.keys():
        # Filter cities :
        city = str(tweetByCity).split("_")[0]
        state = str(tweetByCity).split("_")[1]
        country = str(tweetByCity).split("_")[2]
        if city in listOfCities:
            matrix = pd.DataFrame(tweets[tweetByCity])
            matrix['city'] = city
            matrix['state'] = state
            matrix['country'] = country
            matrixAllTweets = matrixAllTweets.append(matrix, ignore_index=True)

    # Split datetime into date and time
    matrixAllTweets["date"] = [d.date() for d in matrixAllTweets['created_at']]
    matrixAllTweets["time"] = [d.time() for d in matrixAllTweets['created_at']]

    # Filter by a period
    if temporal_period != "all":
        mask = ((matrixAllTweets["date"] >= temporal_period.min()) & (matrixAllTweets["date"] <= temporal_period.max()))
        matrixAllTweets = matrixAllTweets.loc[mask]

    # Compute TF-IDF and TF by state
    extractBiggestTF_allstates = pd.DataFrame()
    extractBiggestTFIDF_allstates = pd.DataFrame()

    if spatial_hiearchy == "country":
        listOfLocalities = listOfCountry
    elif spatial_hiearchy == "state":
        listOfLocalities = listOfStates
    elif spatial_hiearchy == "city":
        listOfLocalities = listOfCities

    for locality in listOfLocalities:
        matrix_by_locality = matrixAllTweets[matrixAllTweets[spatial_hiearchy] == locality]
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=0.001,
            ngram_range=(1, 2),
            token_pattern='[a-zA-Z0-9#@]+',
        )
        # logger.info("Compute TF-IDF on corpus = "+spatial_hiearchy)
        try:
            vectors = vectorizer.fit_transform(matrix_by_locality['tweet'])
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
        except:
            logger.info("Impossible to compute TF-IDF on: "+locality)
            continue
        ## matrixTFIDF
        TFIDFClassical = pd.DataFrame(denselist, columns=feature_names)
        locality_format = locality.replace("/", "_")
        locality_format = locality_format.replace(" ", "_")
        logger.info("saving TF-IDF File: "+path_for_filesaved+"/tfidf_on_"+locality_format+"_corpus.csv")
        TFIDFClassical.to_csv(path_for_filesaved+"/tfidf_on_"+locality_format+"_corpus.csv")
        ## Extract N TOP ranking score
        extractBiggest = TFIDFClassical.max().nlargest(nb_biggest_terms)
        extractBiggest = extractBiggest.to_frame()
        extractBiggest = extractBiggest.reset_index()
        extractBiggest.columns = ['terms', 'score']
        extractBiggest[spatial_hiearchy] = locality
        extractBiggestTFIDF_allstates = extractBiggestTFIDF_allstates.append(extractBiggest, ignore_index=True)

        """
        # Compute TF
        tf = CountVectorizer(
            stop_words='english',
            min_df=2,
            ngram_range=(1,2),
            token_pattern='[a-zA-Z0-9@#]+',
        )
        try:
            tf.fit(matrix_by_locality['tweet'])
            tf_res = tf.transform(matrix_by_locality['tweet'])
            listOfTermsTF = tf.get_feature_names()
            countTerms = tf_res.todense()
        except:# locality does not have enough different term
            logger.info("Impossible to compute TF on: "+locality)
            continue
        ## matrixTF
        TFClassical = pd.DataFrame(countTerms.tolist(), columns=listOfTermsTF)
        ### save in file
        logger.info("saving TF File: "+path_for_filesaved+"/tf_on_"+locality.replace("/", "_")+"_corpus.csv")
        TFClassical.to_csv(path_for_filesaved+"/tf_on_"+locality.replace("/", "_")+"_corpus.csv")
        ## Extract N TOP ranking score
        extractBiggestTF = TFClassical.max().nlargest(nb_biggest_terms)
        extractBiggestTF = extractBiggestTF.to_frame()
        extractBiggestTF = extractBiggestTF.reset_index()
        extractBiggestTF.columns = ['terms', 'score']
        extractBiggestTF[spatial_hiearchy] = locality
        extractBiggestTF_allstates = extractBiggestTF_allstates.append(extractBiggestTF, ignore_index=True)
    """

    logger.info("saving TF and TF-IDF top"+str(nb_biggest_terms)+" biggest score")
    extractBiggestTF_allstates.to_csv(path_for_filesaved+"/TF_BiggestScore_on_"+spatial_hiearchy+"_corpus.csv")
    extractBiggestTFIDF_allstates.to_csv(path_for_filesaved+"/TF-IDF_BiggestScore_on_"+spatial_hiearchy+"_corpus.csv")

def TFIDF_TF_on_whole_corpus(elastic_query_fname, logger, path_for_filesaved="./",
                             temporal_period='all', listOfCities='all'):
    """
    Compute TFIDF and TF from an elastic query file
    1 doc = 1 tweet
    Corpus = on the whole elastic query (with filter out cities that are not in listOfCities

    :param elastic_query_fname: filename and path of the elastic query
    :param logger: logger of the main program
    :param nb_biggest_terms: How many biggest term are to keep. It has to be greater than H-TF-IDF or
        TF-IDF classical on corpus = localité because a lot of temrs have 1.0 has the score
    :param spatial_hiearchy: define the size of the corpus : state or country
    :param temporal_period:
    :param listOfCities: If you want to filter out some cities, you can
    :return:
    """
    # tfidfStartDate = date(2020, 1, 23)
    # tfidfEndDate = date(2020, 1, 30)
    # temporal_period = pd.date_range(tfidfStartDate, tfidfEndDate)
    # listOfCity = ['London', 'Glasgow', 'Belfast', 'Cardiff']
    # listOfState = ["England", "Scotland", "Northern Ireland", "Wales"]

    # Query Elasticsearch to get all tweets from UK
    tweets = elasticsearch_query(elastic_query_fname, logger)
    if listOfCities == 'all':
        listOfCities = []
        listOfStates = []
        listOfCountry = []
        for triple in tweets:
            splitted = triple.split("_")
            listOfCities.append(splitted[0])
            listOfStates.append(splitted[1])
            listOfCountry.append(splitted[2])
        listOfCities = list(set(listOfCities))
        listOfStates = list(set(listOfStates))
        listOfCountry = list(set(listOfCountry))
    # reorganie tweets (dict : tweets by cities) into dataframe (city and date)
    matrixAllTweets = pd.DataFrame()
    for tweetByCity in tweets.keys():
        # Filter cities :
        city = str(tweetByCity).split("_")[0]
        if city in listOfCities:
            matrix = pd.DataFrame(tweets[tweetByCity])
            matrixAllTweets = matrixAllTweets.append(matrix, ignore_index=True)

    # Split datetime into date and time
    matrixAllTweets["date"] = [d.date() for d in matrixAllTweets['created_at']]
    matrixAllTweets["time"] = [d.time() for d in matrixAllTweets['created_at']]

    # Filter by a period
    if temporal_period != "all":
        mask = ((matrixAllTweets["date"] >= temporal_period.min()) & (matrixAllTweets["date"] <= temporal_period.max()))
        matrixAllTweets = matrixAllTweets.loc[mask]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=0.001,
        ngram_range=(1, 2),
        token_pattern='[a-zA-Z0-9#]+', #remove user name, i.e term starting with @ for personnal data issue
    )
    # logger.info("Compute TF-IDF on corpus = "+spatial_hiearchy)
    try:
        vectors = vectorizer.fit_transform(matrixAllTweets['tweet'])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
    except:
        logger.info("Impossible to compute TF-IDF")
        exit(-1)
    ## matrixTFIDF
    TFIDFClassical = pd.DataFrame(denselist, columns=feature_names)
    logger.info("saving TF-IDF File: "+path_for_filesaved+"/tfidf_on_whole_corpus.csv")
    TFIDFClassical.to_csv(path_for_filesaved+"/tfidf_on_whole_corpus.csv")
    ## Extract N TOP ranking score
    extractBiggest = TFIDFClassical.max()
    extractBiggest = extractBiggest[extractBiggest == 1] # we keep only term with high score TF-IDF, i.e 1.0
    extractBiggest = extractBiggest.to_frame()
    extractBiggest = extractBiggest.reset_index()
    extractBiggest.columns = ['terms', 'score']

    logger.info("saving  TF-IDF top"+str(extractBiggest['terms'].size)+" biggest score")
    extractBiggest.to_csv(path_for_filesaved+"/TFIDF_BiggestScore_on_whole_corpus.csv")

def logsetup(log_fname):
    """
    Initiate a logger object :
        - Log in file : collectweets.log
        - also print on screen
    :return: logger object
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(funcName)20s() ::%(message)s')
    now = datetime.now()
    file_handler = RotatingFileHandler(log_fname + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log", 'a', 1000000, 1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    # Only display on screen INFO
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


def t_SNE_bert_embedding_visualization(biggest_score, logger, listOfLocalities="all", spatial_hieararchy="country",
                                       plotname="colored by country", paht2save="./"):
    """
    Plot t-SNE representation of terms by country
    ressources:
        + https://colab.research.google.com/drive/1FmREx0O4BDeogldyN74_7Lur5NeiOVye?usp=sharing#scrollTo=Fbq5MAv0jkft
        + https://github.com/UKPLab/sentence-transformers
    :param biggest_score:
    :param listOfLocalities:
    :param spatial_hieararchy:
    :param plotname:
    :param paht2save:
    :return:
    """
    modelSentenceTransformer = SentenceTransformer('distilbert-base-nli-mean-tokens')


    # filter by localities
    for locality in biggest_score[spatial_hieararchy].unique():
        if locality not in listOfLocalities:
            biggest_score = biggest_score.drop(biggest_score[biggest_score[spatial_hieararchy] == locality].index)

    embeddings = modelSentenceTransformer.encode(biggest_score['terms'].to_list(), show_progress_bar=True)
    # embeddings.tofile(paht2save+"/tsne_bert-embeddings_"+plotname+"_matrix-embeddig")

    modelTSNE = TSNE(n_components=2)  # n_components means the lower dimension
    low_dim_data = modelTSNE.fit_transform(embeddings)

    label_tsne = biggest_score[spatial_hieararchy]

    # Style Plots a bit
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

    plt.rcParams['figure.figsize'] = (20, 14)

    tsne_df = pd.DataFrame(low_dim_data, label_tsne)
    tsne_df.columns = ['x', 'y']
    ax = sns.scatterplot(data=tsne_df, x='x', y='y', hue=tsne_df.index)
    plt.ylim(-100,100)
    plt.xlim(-100, 100)
    ax.set_title('T-SNE BERT Sentence Embeddings for '+plotname)
    plt.savefig(paht2save+"/tsne_bert-embeddings_"+plotname)
    logger.info("file: "+paht2save+"/tsne_bert-embeddings_"+plotname+" has been saved.")
    #plt.show()
    plt.close()

    # Perform kmean clustering
    # num_clusters = 5
    # clustering_model = KMeans(n_clusters=num_clusters)
    # clustering_model.fit(embeddings)
    # cluster_assignment = clustering_model.labels_

    # Normalize the embeddings to unit length
    corpus_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    # clustered_sentences = [[] for i in range(num_clusters)]
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     clustered_sentences[cluster_id].append(biggest_score['terms'].iloc[sentence_id])

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(biggest_score['terms'].iloc[sentence_id])

    #for i, cluster in enumerate(clustered_sentences):
    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        print(cluster)
        print("")

def bert_embedding_filtred(biggest_score, listOfLocalities="all", spatial_hieararchy="country"):
    """
    Retrieve embedding of a matrix of termes (possibility of filtring by a list of locality)
    :param biggest_score: pd.Datraframe with columns : [terms, country/state/city]
    :param listOfLocalities:
    :param spatial_hieararchy:
    :return:
    """
    modelSentenceTransformer = SentenceTransformer('distilbert-base-nli-mean-tokens')
    # filter by localities
    if listOfLocalities != "all":
        for locality in biggest_score[spatial_hieararchy].unique():
            if locality not in listOfLocalities:
                biggest_score = biggest_score.drop(biggest_score[biggest_score[spatial_hieararchy] == locality].index)

    embeddings = modelSentenceTransformer.encode(biggest_score['terms'].to_list(), show_progress_bar=True)
    return embeddings

def similarity_intra_matrix_pairwise(matrix):
    """
    Compute pairwise cosine similarity on the rows of a Matrix and retrieve unique score by pair.
    indeed, cosine_similarity pairwise retrive a matrix with duplication : let's take an exemple :
    Number of terms : 4, cosine similarity :
            w1   w2  w3  w4
            +---+---+----+--+
        w1  | 1 |   |   |   |
        w2  |   | 1 |   |   |
        w3  |   |   | 1 |   |
        w4  |   |   |   | 1 |
            +---+---+----+--+

        (w1, w2) = (w2, w1), so we have to keep only  : (number_of_terms)^2/2 - (number_of_terms)/2
                                                        for nb_term = 4 :
                                                        4*4/2 - 4/2 = 16/2 - 4/2 = 6 => we have 6 unique scores

    :param matrix:
    :return: list of unique similarity score
    """
    similarity = cosine_similarity(sparse.csr_matrix(matrix))
    similarity_1D = np.array([])
    for i, row in enumerate(similarity):
        similarity_1D = np.append(similarity_1D, row[i+1:]) # We remove duplicate pairwise value
    return similarity_1D

def similarity_inter_matrix(matrix1, matrix2):
    """

    :param matrix1:
    :param matrix2:
    :return:
    """
    similarity = 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')
    return similarity

def geocoding_token(biggest, listOfLocality, spatial_hieararchy, logger):
    """
    Find and geocode Spatial entity with OSM data (nominatim)
    Respect terms and use of OSM and Nomitim :
        - Specify a name for the application, Ie.e user agent
        - add delay between each query : min_delay_seconds = 1.
            See : https://geopy.readthedocs.io/en/stable/#module-geopy.extra.rate_limiter
        - define a time out for waiting nomatim answer : to 10 seconds
    :param biggest:
    :return: biggest with geocoding information
    """
    try:
        if listOfLocality != "all":
            for locality in biggest[spatial_hieararchy].unique():
                if locality not in listOfLocality:
                    biggest = biggest.drop(biggest[biggest[spatial_hieararchy] == locality].index)
    except:
        logger.info("could not filter, certainly because there is no spatial hiearchy on biggest score")

    geolocator = Nominatim(user_agent="h-tfidf-evaluation", timeout=10)
    geocoder = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    tqdm.pandas()
    biggest["geocode"] = biggest["terms"].progress_apply(geocoder)
    return biggest

def post_traitement_flood(biggest, logger, spatialLevel, ratio_of_flood=0.5):
    """
    Remove terms from people flooding : return same dataframe with 1 more column : user_flooding
    With default ratio_of_flood : If an twitter.user use a term in more than 50% of occurence of this terms,
       we consider this user is flooding

    :param biggest: File of terms to process
    :param logger:
    :param: spatialLevel : work on Country / State / City
    :param: ratio_of_flood
    :return: return same dataframe with 1 more column : user_flooding
    """
    ratio_of_flood_global = ratio_of_flood
    es_logger.setLevel(logging.WARNING)

    # pre-build elastic query for spatialLevel :
    rest_user_osm_level = ""
    if spatialLevel == "country":
        rest_user_osm_level = "rest_user_osm.country"
    elif spatialLevel == "state":
        rest_user_osm_level = "rest.features.properties.state"
    elif spatialLevel == "city":
        rest_user_osm_level = "rest.features.properties.city"

    def is_an_user_flooding(term, locality):
        client = Elasticsearch("http://localhost:9200")
        index = "twitter"
        # Query :
        ## Retrieve only user name where in full_text = term and rest_user_osm.country = locality
        if term is not np.NAN:
            query = {"_source": "user.name","query":{"bool":{"filter":[{"bool":{"should":[{"match_phrase":{"full_text":term}}],"minimum_should_match":1}},
                                                                       {"bool":{"should":[{"match_phrase":{rest_user_osm_level:locality}}],"minimum_should_match":1}}]}}}
            try:
                result = Elasticsearch.search(client, index=index, body=query)
                list_of_user = []
                if len(result["hits"]["hits"]) != 0:
                    for hit in result["hits"]["hits"]:
                        user = hit["_source"]["user"]["name"]
                        list_of_user.append(user)
                    dict_user_nbtweet = dict(Counter(list_of_user))
                    d = dict((k, v) for k, v in dict_user_nbtweet.items() if v >= (ratio_of_flood_global * len(list_of_user)))
                    if len(d) > 0 : # there is a flood on this term:
                        return 1
                    else:
                        return 0
                else: # not found in ES why ?
                    return "not_in_es"
            except:
                logger.info("There is a trouble with this term: " + str(term))
                return np.NAN
        else:
            return 0

    logger.debug("start remove terms if they coming from a flooding user, ie, terms in "+str(ratio_of_flood_global*100)+"% of tweets from an unique user over tweets with this words")
    tqdm.pandas()
    biggest["user_flooding"] = biggest.progress_apply(lambda t: is_an_user_flooding(t.terms, t[spatialLevel]), axis=1)
    return biggest



if __name__ == '__main__':
    # Workflow parameters :
    ## Rebuild H-TFIDF (with Matrix Occurence)
    build_htfidf = True
    ## eval 1 : Comparison with classical TF-IDf
    build_classical_tfidf = False
    ## evla 2 : Use word_embedding with t-SNE
    build_tsne = False
    build_tsne_spatial_level = "country"
    ## eval 3 : Use word_embedding with box plot to show disparity
    build_boxplot = False
    build_boxplot_spatial_level = "country"
    ## post-traitement 1 : geocode term
    build_posttraitement_geocode = False
    ## post-trautement 2 : remove terms form a flooding user
    build_posttraitement_flooding = True
    build_posttraitement_flooding_spatial_levels = ['country', 'state', 'city']

    # Global parameters :
    ## Path to results :
    f_path_result = "elasticsearch/analyse/nldb21/results/4thfeb"
    ## Spatial level hierarchie :
    spatialLevels = ['country', 'state', 'city']
    ## Time level hierarchie :
    timeLevel = "week"
    ## List of country to work on :
    listOfLocalities = ["France", "Deutschland", "España", "Italia", "United Kingdom"]
    ## elastic query :
    query_fname = "elasticsearch/analyse/nldb21/elastic-query/nldb21_europeBySpatialExtent_en_4thweekFeb.txt"

    # initialize a logger :
    log_fname = "elasticsearch/analyse/nldb21/logs/nldb21_"
    logger = logsetup(log_fname)
    logger.info("H-TFIDF expirements starts")


    if build_htfidf:
        # start the elastic query
        query = open(query_fname, "r").read()
        logger.debug("elasticsearch : start quering")
        tweetsByCityAndDate = elasticsearch_query(query_fname, logger)
        logger.debug("elasticsearch : stop quering")

        # Build a matrix of occurence for each terms in document aggregate by city and day
        ## prepare tree for file in commun for all spatial level :
        f_path_result_common = f_path_result+"/common"
        if not os.path.exists(f_path_result_common):
            os.makedirs(f_path_result_common)
        ## Define file path
        matrixAggDay_fpath = f_path_result_common + "/matrixAggDay.csv"
        matrixOccurence_fpath = f_path_result_common + "/matrixOccurence.csv"
        logger.debug("Build matrix of occurence : start")
        matrixOccurence = matrixOccurenceBuilder(tweetsByCityAndDate, matrixAggDay_fpath, matrixOccurence_fpath, logger)
        logger.debug("Build matrix of occurence : stop")
        ## import matrixOccurence if you don't want to re-build it
        # matrixOccurence = pd.read_csv('elasticsearch/analyse/matrixOccurence.csv', index_col=0)

        for spatialLevel in spatialLevels:
            logger.info("H-TFIDF on: "+spatialLevel)
            f_path_result_level = f_path_result+"/"+spatialLevel
            if not os.path.exists(f_path_result_level):
                os.makedirs(f_path_result_level)
            ## Compute H-TFIDF
            matrixHTFIDF_fname = f_path_result_level + "/matrix_H-TFIDF.csv"
            biggestHTFIDFscore_fname = f_path_result_level + "/h-tfidf-Biggest-score.csv"
            logger.debug("H-TFIDF : start to compute")
            HTFIDF(matrixOcc=matrixOccurence,
                   matrixHTFIDF_fname=matrixHTFIDF_fname,
                   biggestHTFIDFscore_fname=biggestHTFIDFscore_fname,
                   spatialLevel=spatialLevel,
                   temporalLevel=timeLevel,
                   )
        logger.info("H-TFIDF : stop to compute for all spatial levels")

    ## Comparison with TF-IDF
    f_path_result_tfidf = f_path_result + "/tf-idf-classical"
    f_path_result_tfidf_by_locality = f_path_result_tfidf + "/tfidf-tf-corpus-country"
    if build_classical_tfidf :
        if not os.path.exists(f_path_result_tfidf):
            os.makedirs(f_path_result_tfidf)
        if not os.path.exists(f_path_result_tfidf_by_locality):
            os.makedirs(f_path_result_tfidf_by_locality)
        ### On whole corpus
        TFIDF_TF_on_whole_corpus(elastic_query_fname=query_fname,
                                 logger=logger,
                                 path_for_filesaved=f_path_result_tfidf)
        ### By Country
        TFIDF_TF_with_corpus_state(elastic_query_fname=query_fname,
                                   logger=logger,
                                   nb_biggest_terms=500,
                                   path_for_filesaved=f_path_result_tfidf_by_locality,
                                   spatial_hiearchy="country",
                                   temporal_period='all')
    if build_tsne :
        f_path_result_tsne = f_path_result+"/tsne"
        biggest_TFIDF_country = pd.read_csv(f_path_result_tfidf_by_locality+"TF-IDF_BiggestScore_on_country_corpus.csv", index_col=0)
        biggest_TFIDF_whole = pd.read_csv(f_path_result_tfidf+"TFIDF_BiggestScore_on_whole_corpus.csv")
        biggest_H_TFIDF = pd.read_csv(f_path_result+"/"+build_tsne_spatial_level+'/h-tfidf-Biggest-score.csv', index_col=0)
        # t_SNE visulation
        t_SNE_bert_embedding_visualization(biggest_TFIDF_country, logger, listOfLocalities=listOfLocalities,
                                           plotname="TF-IDF on corpus by Country",
                                           paht2save=f_path_result_tsne)
        t_SNE_bert_embedding_visualization(biggest_H_TFIDF, logger, listOfLocalities=listOfLocalities,
                                           plotname="H-TFIDF", paht2save=f_path_result_tsne)

    if build_boxplot :
        # dir path to save :
        f_path_result_boxplot = f_path_result+"/pairwise-similarity-boxplot"
        if not os.path.exists(f_path_result_boxplot):
            os.makedirs(f_path_result_boxplot)
        # open result from mesures :
        biggest_TFIDF_country = pd.read_csv(f_path_result_tfidf_by_locality+"/TF-IDF_BiggestScore_on_country_corpus.csv", index_col=0)
        biggest_TFIDF_whole = pd.read_csv(f_path_result_tfidf+"/TFIDF_BiggestScore_on_whole_corpus.csv")
        biggest_H_TFIDF = pd.read_csv(f_path_result+"/"+build_boxplot_spatial_level+'/h-tfidf-Biggest-score.csv', index_col=0)
        # Retrieve embedding :
        htfidf_embeddings = bert_embedding_filtred(biggest_H_TFIDF, listOfLocalities=listOfLocalities)
        tfidf_country_embeddings = bert_embedding_filtred(biggest_TFIDF_country, listOfLocalities=listOfLocalities)
        tfidf_whole_embeddings = bert_embedding_filtred(biggest_TFIDF_whole)
        # Compute similarity :
        ## Distribution of similarities between terms extracted from a measure
        htidf_similarity = similarity_intra_matrix_pairwise(htfidf_embeddings)
        tfidf_country_similarity = similarity_intra_matrix_pairwise(tfidf_country_embeddings)
        tfidf_whole_similarity = similarity_intra_matrix_pairwise(tfidf_whole_embeddings)

        plt.subplot(131)
        plt.boxplot(htidf_similarity)
        plt.title("H-TFIDF")
        plt.ylim(0,1)
        plt.subplot(132)
        plt.boxplot(tfidf_country_similarity)
        plt.title("TFIDF with corpus by country")
        plt.ylim(0, 1)
        plt.subplot(133)
        plt.boxplot(tfidf_whole_similarity)
        plt.title("TFIDF on the whole corpus")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.suptitle("Distribution of similarity values among the extracted terms pairs of a measure")
        plt.savefig(f_path_result_boxplot+"/pairwise-similarity-boxplot.png")
        # plt.show()
        plt.close()
        ## Distribution of similarities between the terms of a country extracted from a measure
        ### H-TFIDF
        fig2, axs2 = plt.subplots(1, 5)
        for i, country in enumerate(listOfLocalities):
            axs2[i].boxplot(similarity_intra_matrix_pairwise(htfidf_embeddings[i*500:(i+1)*500-1]))
            axs2[i].set_title(country)
            axs2[i].set_ylim(0, 1)
        fig2.suptitle("Distribution of similarity by pairs for H-TF-IDF")
        plt.savefig(f_path_result_boxplot + "/pairwise-similarity-boxplot_HTFIDF-country.png")
        # plt.show()
        plt.close(fig2)
        ### TF-IDF by corpus = country
        fig3, axs3 = plt.subplots(1, 5)
        for i, country in enumerate(listOfLocalities):
            axs3[i].boxplot(similarity_intra_matrix_pairwise(tfidf_country_embeddings[i*500:(i+1)*500-1]))
            axs3[i].set_title(country)
            axs3[i].set_ylim(0, 1)
        fig3.suptitle("Distribution of similarity by pairs for TF-IDF focus on each country")
        plt.savefig(f_path_result_boxplot + "/pairwise-similarity-boxplot_TFIDF-country.png")
        # plt.show()
        plt.close(fig3)
        ## Distribution of similarities between the set of terms of 2 measures
        ### H-TF-IDF with TF-IDF on whole corpus and TF-IDF country with TF-IDF on whole corpus
        fig_compare_TFIDF_whole, ax4 = plt.subplots(1,2)
        similarity_between_htfidf_tfidf_whole = similarity_inter_matrix(htfidf_embeddings, tfidf_whole_embeddings)
        similarity_between_tfidfcountry_tfidf_whole = similarity_inter_matrix(tfidf_country_embeddings, tfidf_whole_embeddings)
        similarity_between_htfidf_tfidf_whole_1D = np.array([])
        similarity_between_tfidfcountry_tfidf_whole_1D = np.array([])
        for i, row in enumerate(similarity_between_htfidf_tfidf_whole):
            similarity_between_htfidf_tfidf_whole_1D = np.append(similarity_between_htfidf_tfidf_whole_1D, row[i+1:]) # We remove duplicate pairwise value
        for i, row in enumerate(similarity_between_tfidfcountry_tfidf_whole):
            similarity_between_tfidfcountry_tfidf_whole_1D = np.append(similarity_between_tfidfcountry_tfidf_whole_1D,
                                                                 row[i + 1:])
        ax4[0].boxplot(similarity_between_htfidf_tfidf_whole_1D)
        ax4[0].set_ylim(0, 1)
        ax4[0].set_title("H-TFIDF")
        ax4[1].boxplot(similarity_between_tfidfcountry_tfidf_whole_1D)
        ax4[1].set_ylim(0, 1)
        ax4[1].set_title("TFIDF on country")
        fig_compare_TFIDF_whole.suptitle("Distribution of similarity between H-TFIDF and TF-IDF on whole corpus")
        plt.savefig(f_path_result_boxplot + "/pairwise-similarity-boxplot_between_TFIDF-whole.png")
        # plt.show()
        plt.close(fig_compare_TFIDF_whole)
        ## Distribution of similarities between sub-set terms by country compared by country pair

    if build_posttraitement_geocode:
        # Geocode terms :
        ## Comments : over geocode even on non spatial entities
        spatial_level = "country"
        listOfLocalities = ["France", "Deutschland", "España", "Italia", "United Kingdom"]
        f_path_result = "elasticsearch/analyse/nldb21/results/4thfeb_country"
        biggest_TFIDF_country = pd.read_csv(
            f_path_result+"/tfidf-tf-corpus-country/TF-IDF_BiggestScore_on_"+spatial_level+"_corpus.csv", index_col=0)
        biggest_TFIDF_whole = pd.read_csv(f_path_result+"/TFIDF_BiggestScore_on_whole_corpus.csv")
        biggest_H_TFIDF = pd.read_csv(f_path_result+'/h-tfidf-Biggest-score.csv', index_col=0)
        biggest_H_TFIDF_gepocode = geocoding_token(biggest_H_TFIDF,
                                                   listOfLocality=listOfLocalities,
                                                   spatial_hieararchy=spatial_level,
                                                   logger=logger)
        biggest_H_TFIDF_gepocode.to_csv(f_path_result+"/h-tfidf-Biggest-score-geocode.csv")
        biggest_TFIDF_country_gepocode = geocoding_token(biggest_TFIDF_country,
                                                   listOfLocality=listOfLocalities,
                                                   spatial_hieararchy=spatial_level,
                                                         logger=logger)
        biggest_TFIDF_country_gepocode.to_csv(f_path_result+"/TF-IDF_BiggestScore_on_"+spatial_level+"_corpus_geocode.csv")
        biggest_TFIDF_whole_gepocode = geocoding_token(biggest_TFIDF_whole,
                                                   listOfLocality=listOfLocalities,
                                                   spatial_hieararchy=spatial_level,
                                                       logger=logger)
        biggest_TFIDF_whole_gepocode.to_csv(f_path_result+"/TFIDF_BiggestScore_on_whole_corpus_geocode.csv")


    if build_posttraitement_flooding:
        # Post traitement : remove terms coming from user who flood
        for spatial_level_flood in build_posttraitement_flooding_spatial_levels:
            logger.info("post-traitement flooding on: " + spatial_level_flood)
            f_path_result_flood = f_path_result + "/" + spatial_level_flood
            biggest_H_TFIDF = pd.read_csv(f_path_result_flood + '/h-tfidf-Biggest-score.csv', index_col=0)
            biggest_H_TFIDF_with_flood = post_traitement_flood(biggest_H_TFIDF, logger, spatialLevel=spatial_level_flood)
            biggest_H_TFIDF_with_flood.to_csv(f_path_result_flood + "/h-tfidf-Biggest-score-flooding.csv")

    logger.info("H-TFIDF expirements stops")
