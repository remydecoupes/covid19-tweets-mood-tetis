#!/usr/bin/env python

"""
Merge biotex results from 30k tweets per files
"""
import pandas as pd
from pathlib import Path
import os
import numpy as np

biotexparams = ['ftfidfc-all', 'ftfidfc-multi', 'c-value-all', 'c-value-multi']

def mergeBiotex(biotexResultDir, mergeResultDir):
    columnsName = ['term', 'max', 'sum', 'occurence', 'average', 'ulms']
    for param in biotexparams:
        dfTerms = pd.DataFrame(columns=columnsName)
        i = 0
        biotexResultDirParam = biotexResultDir.joinpath(param)
        for file in biotexResultDirParam.glob("biotex*"):
            i += 1
            dfToMerge = pd.read_csv(file, sep='\t')
            dfToMerge.columns = ['term','ulms'+str(i), 'score'+str(i)]
            dfTerms = dfTerms.merge(dfToMerge, on='term', how='outer') # outer : union of keys from both frames,
                # similar to a SQL full outer join; sort keys lexicographically.
                # Default is inner : intersection
            dfTerms["max"] = dfTerms[["max", 'score'+str(i)]].max(axis=1) # axis = 1 <=> column
            dfTerms["sum"] = dfTerms[["sum", 'score'+str(i)]].sum(axis=1)
            ## Average
            # occurence number for each term
            print(i)
            dfTerms['occurence'].fillna(0, inplace=True) # transform NA to 0
            dfTerms.loc[dfTerms['term'].isin(dfToMerge['term']), 'occurence'] += 1
            dfTerms["average"] = dfTerms["sum"] / dfTerms['occurence']
            ## ULMS
            dfTerms['ulms'+str(i)] = dfTerms['ulms'+str(i)].astype(bool)
            dfTerms["ulms"] = dfTerms["ulms"] | dfTerms['ulms'+str(i)]
            # delete row after aggregation
            dfTerms = dfTerms.drop(['score'+str(i), 'ulms'+str(i)], 1) # ,1 : axis : column
        dfTerms.to_csv(mergeResultDir.joinpath("merge30ktweets-english-"+param+".csv"))
        print("save file: "+str(mergeResultDir.joinpath("merge30ktweets-english-"+param+".csv")))
        #faire les sort : max, average et sum
        # commenter les résultats

def rankMergeResult(mergeResultDir):
    """

    :param mergeResultDir:
    :return:
    """
    rankedMeasures = ['max', 'sum', 'average']
    dfcompare = pd.DataFrame()
    for file in mergeResultDir.glob("merge*"):
        df = pd.read_csv(file)
        for measure in rankedMeasures:
            df.sort_values(by=measure, inplace=True, ascending=False)
            # print("Biotex Param: "+str(file.name).replace("merge30ktweets-english-","")+" | measure: "+measure)
            # print(df['term'].head(n=20).to_string(index=False))
            dfcompare[str(file.name).replace("merge30ktweets-english-", "").replace(".csv", "") + "_" + measure] = \
                df['term'].head(n=20).values
    column_order = ['ftfidfc-multi_max', 'ftfidfc-all_max', 'ftfidfc-multi_average', 'ftfidfc-all_average',
                    'ftfidfc-multi_sum', 'ftfidfc-all_sum', 'c-value-multi_max', 'c-value-all_max',
                    'c-value-multi_average', 'c-value-multi_average', 'c-value-multi_sum', 'c-value-multi_sum']
    dfcompare[column_order].to_csv(str(mergeResultDir)+"/compareparam.csv")




if __name__ == '__main__':
    print("begin")
    biotexResultDir = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults/subdividedcorpus')
    mergeResultDir = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults/subdividedcorpus/merge')
    # mergeBiotex(biotexResultDir, mergeResultDir)
    rankMergeResult(mergeResultDir)
    print("end")
