#!/usr/bin/env python

"""
Transform JSON fastr into CSV (to show results)
"""
from pathlib import Path
import json
import pandas as pd

if __name__ == '__main__':
    print("begin FASTR: Json to CSV")
    fastrJson = \
        Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/fastr/driven_extraction_version_300K.json')
    fastrCSV = \
        Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/fastr/driven_extraction_version_300K.csv')

    ## Read Variants from FASTR as a dict
    with open(fastrJson) as json_file:
        fastr = json.load(json_file)
    csv = pd.DataFrame()

    for key, value in fastr.items():
        csv[key] = pd.Series(value[0]) # because length are not egual
    csv = csv.fillna("") # remplace NAN by ""

    csv.to_csv(fastrCSV)
    print("begin FASTR: Json to CSV")