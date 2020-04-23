#!/usr/bin/env python

"""
Merge biotex results from 30k tweets per files
"""
import pandas as pd
from pathlib import Path
import json

biotexparams = ['ftfidfc-all', 'ftfidfc-multi', 'c-value-all', 'c-value-multi']


def mergeBiotex(biotexResultDir, mergeResultDir):
    columnsName = ['term', 'max', 'sum', 'occurence', 'average', 'umls']
    for param in biotexparams:
        dfTerms = pd.DataFrame(columns=columnsName)
        i = 0
        biotexResultDirParam = biotexResultDir.joinpath(param)
        for file in biotexResultDirParam.glob("fastr*"):
            i += 1
            dfToMerge = pd.read_csv(file, sep=',')
            dfToMerge.columns = [i, 'term','umls'+str(i), 'score'+str(i)]
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
            # delete row after aggregation
            dfTerms = dfTerms.drop([i, 'score'+str(i), 'umls'+str(i)], 1) # ,1 : axis : column
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

def rankMergeResult(mergeResultDir):
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
    column_order = ['ftfidfc-all_average', 'ftfidfc-all_average_UMLS','ftfidfc-all_mutltiExtracted_average',
                    'ftfidfc-all_mutltiExtracted_average_UMLS', 'c-value-all_average', 'c-value-all_average_UMLS',
                    'c-value-all_mutltiExtracted_average', 'c-value-all_mutltiExtracted_average_UMLS']
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

            # Start E3 : extract multi terms from other
            dfextractMulti = pd.DataFrame()
            ## build a new column with only multi terms (terms which contains a space " ")
            dfextractMulti[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") +
                      "_mutltiExtracted_"+ measure] = df[df['term'].str.contains(" ")]['term'].values
            ## builc column for UMLS
            dfextractMulti[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") +
                           "_mutltiExtracted_" + measure + '_UMLS'] = df[df['term'].str.contains(" ")]['umls'].values
            ## Then concate with the previous. We could not add the column because of his inferior length
            dfcompare = pd.concat([dfcompare, dfextractMulti], axis=1)
            # end of E3

    dfcompare[column_order].head(n=nbTerms).to_csv(str(mergeResultDir) + "/compareparam.csv")



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



if __name__ == '__main__':
    print("begin")
    biotexResultDir = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults/subdividedcorpus')
    mergeResultDir = \
        Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults/subdividedcorpus/merge')
    fastrVariants = \
        Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/fastr/driven_extraction_version_300K.json')
    print("start FASTR")
    fastrOnBiotexResult(biotexResultDir, fastrVariants)
    print("start Merge")
    mergeBiotex(biotexResultDir, mergeResultDir)
    print("start Ranked merge")
    rankMergeResult(mergeResultDir)
    print("end")
