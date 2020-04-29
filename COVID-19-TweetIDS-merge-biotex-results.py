#!/usr/bin/env python

"""
Merge biotex results from 30k tweets per files
"""
import pandas as pd
from pathlib import Path
import json
# SentiWordNet
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
# End Of SentiWordNet
import matplotlib.pyplot as plt



biotexparams = ['ftfidfc-all', 'ftfidfc-multi', 'c-value-all', 'c-value-multi']


def mergeBiotex(biotexResultDir, mergeResultDir):
    columnsName = ['term', 'max', 'sum', 'occurence', 'average', 'umls', 'fastr']
    for param in biotexparams:
        dfTerms = pd.DataFrame(columns=columnsName)
        i = 0
        biotexResultDirParam = biotexResultDir.joinpath(param)
        for file in biotexResultDirParam.glob("fastr*"):
            i += 1
            dfToMerge = pd.read_csv(file, sep=',')
            dfToMerge.columns = [i, 'term','umls'+str(i), 'score'+str(i), 'fastr'+str(i)]
            dfTerms = dfTerms.merge(dfToMerge, on='term', how='outer') # outer : union of keys from both frames,
                # similar to a SQL full outer join; sort keys lexicographically.
                # Default is inner : intersection
            dfTerms["max"] = dfTerms[["max", 'score'+str(i)]].max(axis=1) # axis = 1 <=> column
            dfTerms["sum"] = dfTerms[["sum", 'score'+str(i)]].sum(axis=1)
            ## Average
            # occurence number for each term
            # print(i)
            dfTerms['occurence'].fillna(0, inplace=True) # transform NA to 0
            dfTerms.loc[dfTerms['term'].isin(dfToMerge['term']), 'occurence'] += 1
            dfTerms["average"] = dfTerms["sum"] / dfTerms['occurence']
            ## umls
            dfTerms['umls'+str(i)] = dfTerms['umls'+str(i)].astype(bool)
            dfTerms["umls"] = dfTerms["umls"] | dfTerms['umls'+str(i)]
            ## fastr
            ### tricky tip: replace empty value from fastr by values of fastr-i (some are still empty! Doesn't matter!)
            dfTerms['fastr'].fillna(dfTerms['fastr'+str(i)], inplace=True)
            # delete row after aggregation
            dfTerms = dfTerms.drop([i, 'score'+str(i), 'umls'+str(i), 'fastr'+str(i)], 1) # ,1 : axis : column
        dfTerms.to_csv(mergeResultDir.joinpath("merge30ktweets-english-"+param+".csv"))
        print("save file: "+str(mergeResultDir.joinpath("merge30ktweets-english-"+param+".csv")))
        #faire les sort : max, average et sum
        # commenter les résultats

def cleanMergeResult(df):
    """
    Clean some noise from biotex as
        - ##########end##########
    :param df : a dataframe to clean
    :return: df : a clean dataframe
    """
    df['term'].fillna("", inplace=True)
    #print(df.loc[df['term'].str.contains('##########end##########', case=False)])
    toDelete = df.loc[df['term'].str.contains('##########end##########', case=False)].index
    if not toDelete.empty: # Do we have to delete something ?
        df.drop(toDelete, inplace=True)
        #print(df.head(n=20))
    return df

def rankMergeResult(mergeResultDir, rankedfilename):
    """
    This function rank biotex merged results : MAX, SUM, Average on score from initial biotex
    Modification :
        - E1 : After meeting 2020-04-15 : we decided to give up on multi-term and work only on all (as biotex params)
        - E2 : Clean up results from biotex (remove #######end#####)
        - E3 : Corroborate with E1 : extract multi terms from E1 (with all as biotex params)
        - E6 : Measur post ranking : AVG
    :param mergeResultDir:
    :return:
    """
    # Comment since E1
    # column_order = ['ftfidfc-multi_max', 'ftfidfc-all_max', 'ftfidfc-multi_average', 'ftfidfc-all_average',
    #                 'ftfidfc-multi_sum', 'ftfidfc-all_sum', 'c-value-multi_max', 'c-value-all_max',
    #                 'c-value-multi_average', 'c-value-all_average', 'c-value-multi_sum', 'c-value-all_sum']
    # End of comment since E1
    # E6 measure : AVG
    # rankedMeasures = ['max', 'sum', 'average']
    # column_order = ['ftfidfc-all_max', 'ftfidfc-all_mutltiExtracted_max', 'ftfidfc-all_average',
    #                 'ftfidfc-all_mutltiExtracted_average', 'ftfidfc-all_sum', 'ftfidfc-all_mutltiExtracted_sum',
    #                 'c-value-all_max', 'c-value-all_mutltiExtracted_max', 'c-value-all_average',
    #                 'c-value-all_mutltiExtracted_average', 'c-value-all_sum', 'c-value-all_mutltiExtracted_sum']
    rankedMeasures = ['average']
    column_order = ['ftfidfc-all_average', 'ftfidfc-all_average_UMLS', 'ftfidfc-all_average_fastr',
                    'ftfidfc-all_mutltiExtracted_average', 'ftfidfc-all_mutltiExtracted_average_UMLS',
                    'ftfidfc-all_mutltiExtracted_average_fastr', 'c-value-all_average', 'c-value-all_average_UMLS',
                    'c-value-all_average_fastr', 'c-value-all_mutltiExtracted_average',
                    'c-value-all_mutltiExtracted_average_UMLS', 'c-value-all_mutltiExtracted_average_fastr']
    dfcompare = pd.DataFrame()
    nbTerms = 100

    # for file in mergeResultDir.glob("merge*"): #since E1
    for file in mergeResultDir.glob("*all.csv"):
        df = cleanMergeResult(pd.read_csv(file)) # clean up acocrding to E2
        for measure in rankedMeasures:
            df.sort_values(by=measure, inplace=True, ascending=False)
            # build a new column with a name extracted from the file. It contains Measure F-TFIDF-C or C-value
            # All or multi terms from biotex
            # and the new ranking measure (Max, sum, average) introduce by this function
            dfcompare[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") + "_" + measure] = \
                df['term'].values
            # add UMLS
            dfcompare[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") + "_" + measure +
                      '_UMLS']= df['umls'].values
            # add Fastr
            dfcompare[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") + "_" + measure +
                      '_fastr'] = df['fastr'].values

            # Start E3 : extract multi terms from other
            dfextractMulti = pd.DataFrame()
            ## build a new column with only multi terms (terms which contains a space " ")
            dfextractMulti[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") +
                      "_mutltiExtracted_"+ measure] = df[df['term'].str.contains(" ")]['term'].values
            ## builc column for UMLS
            dfextractMulti[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") +
                           "_mutltiExtracted_" + measure + '_UMLS'] = df[df['term'].str.contains(" ")]['umls'].values
            ## builc column for fastr
            dfextractMulti[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") +
                           "_mutltiExtracted_" + measure + '_fastr'] = df[df['term'].str.contains(" ")]['fastr'].values
            ## Then concate with the previous. We could not add the column because of his inferior length
            dfcompare = pd.concat([dfcompare, dfextractMulti], axis=1)
            # end of E3

    #dfcompare[column_order].head(n=nbTerms).to_csv(str(mergeResultDir) + "/" + rankedfilename)
    dfcompare[column_order].head(n=nbTerms).to_csv(rankedfilename)



def fastrOnBiotexResult(biotexResultDir,fastrvariants):
    """
    - Wrong : E4 : remove Fastr variants terms : Except for the 1rst term, delete all variants fast
        terms from biotex ranking => Wrong
    - E5 : don't remove variants but flag them with their commun term
    Save files with fastr
    :param biotexResultDir: biotex results
    :param fastrvariants: files containing variants from FASTR algo corresponding to this corpus
    :return:
    """
    ## Read Variants from FASTR as a dict
    with open(fastrvariants) as json_file:
        fastr = json.load(json_file)
    ## Open biotex results
    for param in biotexparams:
        biotexResultDirParam = biotexResultDir.joinpath(param)
        for file in biotexResultDirParam.glob("biotex*"):
            df = pd.read_csv(file, sep='\t')
            df.columns = ['term', 'umls', 'score']
            df['fastr'] = ""
            #print(file.name)
            ## browse on variant
            for term in fastr:
                i = 0
                for variant in fastr[term][0]:
                    id = df[df['term'] == variant]
                    if not id.empty:
                        if i > 0:
                            # E5 : don't remove variant but flag them
                            # df.drop(id.index, inplace=True)
                            df.loc[id.index, 'fastr'] = term
                        i += 1
            df.to_csv(biotexResultDirParam.joinpath("fastr"+file.name))

def sentiwordnet(rankmergeresult, resultfile, mergeResultDir):
    """
    Add a positif or negatif tag to extracted terms using WordNet and sentiWordnet
    Token are tansformed indo their lemma form with WordNet corpus and PoS
    Only 1rst wordnet meaning are used

    4 Plots are saved corresponding to the F-TFIDF-C & C-value for all and multiterm
    :param rankmergeresult: file
    :param resultfile: file to save
    :param mergeResultDir: rep to save plots
    :return:
    """

    def penn_to_wn(tag):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None
    # from nltk.stem import WordNetLemmatizer
    # lemmatizer = WordNetLemmatizer()

    def get_sentiment(word, tag):
        """
        From : https://stackoverflow.com/questions/38263039/sentiwordnet-scoring-with-python
        returns list of pos neg and objective score. But returns empty list if not present in senti wordnet.
        """
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            return []
        # Lemmatization : Canonical lexical form (better -> good, walking -> walk, was -> be)
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            return []
        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            return []
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        return [swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]

    # ps = PorterStemmer()
    df = pd.read_csv(rankmergeresult, index_col=0)
    for columns in df:
        if not "UMLS" in columns and not "fastr" in columns:
            sentimentsList = []
            for term in df[columns]:
                try:
                    # for multi-gram we use sum of negativ and positiv socre
                    sumpolaritypos = 0
                    sumpolarityneg = 0
                    for token in word_tokenize(term):
                        # get sentiment as a tuple (positif, negatif, objectif)
                        sentiment = get_sentiment(token, pos_tag(word_tokenize(token))[0][1])
                        sumpolaritypos += sentiment[0]
                        sumpolarityneg += sentiment[1]
                    # compute sentiment polarity
                    if sumpolaritypos > sumpolarityneg:
                        polarity = "pos"
                    elif sumpolaritypos == 0 and sumpolarityneg == 0:
                        polarity = "neutral"
                    else :
                        polarity = "neg"
                except: # for special character
                    # print(term)
                    polarity = float('nan')
                    sumpolaritypos = float('nan')
                # sentimentsList.append(polarity)
                # compute float sentiment : Pos - neg
                sentimentsList.append(sumpolaritypos - sumpolarityneg)
            df[columns+"_sentiment_polarity"] = pd.Series(sentimentsList)

    # Re-sorted columns order by their names :
    df = df.reindex(sorted(df.columns), axis=1)
    # save to file
    df.to_csv(resultfile, index=False)
    # plot
    ## Select only senti column
    senticolumn = [col for col in df.columns if 'sentiment' in col]
    sentidf = df[senticolumn]
    ## Create a column with index
    sentidf.reset_index(inplace=True)
    for col in senticolumn:
        ## Modify x axis
        ax = plt.gca()
        ### Annote x axis : Give a text (here is our term) for each (x,y) with x : index, y : sentiwordnet value
        termRelatedtoCol = str(col).replace("_sentiment_polarity","")
        for i, term in enumerate(df[termRelatedtoCol]):
            # print(str(sentidf.loc[i]['index'])+", "+str(sentidf.loc[i]['ftfidfc-all_average_sentiment_polarity'])+": "+term)
            ax.annotate(str(term), (sentidf.loc[i]['index'], sentidf.loc[i][col]),
                        rotation=45, )
        ### Inverse order of x axis
        ax.invert_xaxis()
        axsub = sentidf.plot(
            kind='scatter',
            title='Term distribution over lexical sentiment analysis for '+termRelatedtoCol,
            x= 'index',
            ax=ax, # invert x axis from 100 to 1
            y=col
        )
        ### Change x label
        axsub.set_xlabel("Rank Position")
        ### Give y axis boudadry to -1, 1 to have the whole scale (and for annotation goes not out the boundaries)
        axsub.set_ylim(-1,1)

        plt.savefig(str(mergeResultDir) + "/fig_" + col)
        plt.show()

def statsOnRankedTerm(rankedfilename):
    """
    Build statistic on ranked terms :
        - Nb of terms in UMLS for each ranked param (C-value & F-TFIDF for All and multi extracted term)
        - Commun term between C-value and F-TFIDF-C
    :param rankedfilename:
    :return:
    """
    df = pd.read_csv(rankedfilename, index_col=0)
    dfstatNbUMLS = pd.DataFrame()
    commonTermAll = commonMultiTerm = 0
    # count UMLS term in ranked term
    for measure in df:
        if not "UMLS" in measure and not "fastr" in measure: #only browse on ranked param
            # count UMLS term in ranked term : sum index where this below condition is true :
            dfstatNbUMLS[measure] = pd.Series((df[measure+"_UMLS"] == True).sum())
    # Count Common term
    try:
        ## for ALL terms
        for value in df['ftfidfc-all_average']:
            if (df['c-value-all_average'] == value).sum():
                commonTermAll += 1
        ## for multi terms extracted
        for value in df['ftfidfc-all_mutltiExtracted_average']:
            if (df['c-value-all_mutltiExtracted_average'] == value).sum():
                commonMultiTerm += 1
    except:
        print("Likely a trouble with column name")
        commonTermAll = commonMultiTerm = float('nan')
    # Print results:
    print("Terms in UMLS by measure: ")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dfstatNbUMLS)
    print("Commun term with C-value and F-TFIDF-C for all terms:")
    print(commonTermAll)
    print("Commun term with C-value and F-TFIDF-C for multi terms extracted:")
    print(commonMultiTerm)






if __name__ == '__main__':
    print("begin")
    biotexResultDir = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults/subdividedcorpus')
    mergeResultDir = \
        Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults/subdividedcorpus/merge')
    fastrVariants = \
        Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/fastr/driven_extraction_version_300K.json')
    rankedfilename = str(mergeResultDir) + "/compareparam.csv"
    sentionrankedfilename = str(mergeResultDir) + "/compareparamWithSenti.csv"
    # print("start FASTR")
    # fastrOnBiotexResult(biotexResultDir, fastrVariants)
    # print("start Merge")
    # mergeBiotex(biotexResultDir, mergeResultDir)
    # print("start Ranked merge")
    # rankMergeResult(mergeResultDir, rankedfilename)
    # print("start sentiWornNet")
    # sentiwordnet(rankedfilename, sentionrankedfilename, mergeResultDir)
    print("Stat on ranked term")
    statsOnRankedTerm(rankedfilename)
    print("end")
