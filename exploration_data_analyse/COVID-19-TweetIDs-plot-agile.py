#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

if __name__ == '__main__':
    # Generate wordcloud for Greece
    build_wordcloud = False
    if build_wordcloud:
        dir = "elasticsearch/analyse/nldb21/results/feb_idf_square_week/country"
        dir_out = "elasticsearch/analyse/nldb21/results/feb_idf_square_week/country/wordcloud"
        list_of_greece_name = ["Greece", "Ελλάς", "Ἑλλάς"]
        list_of_greece_name = ["Greece", "Ἑλλάς"]
        list_of_greece_week = ["2020-02-02", "2020-02-09", "2020-02-16", "2020-02-23"]
        greece_df = pd.read_csv(dir + "/h-tfidf-Biggest-score-flooding.csv")
        greece_df = greece_df[greece_df["user_flooding"] == "0"]
        for name in list_of_greece_name:
            greece_name_df = greece_df[greece_df["country"] == name]
            for week in greece_name_df["date"].unique():
                greece_name_w_df = greece_name_df[greece_name_df["date"] == week]
                # list_of_terms = greece_name_w_df["terms"]
                terms_ranks = {}
                # using wordcloud with frequency : font size depanding of the H-TFIDF ranked
                for rank, term in enumerate(greece_name_w_df["terms"]):
                    terms_ranks[term] = 500 - rank
                try:
                    wordcloud = WordCloud(background_color="white", width=1600, height=800)
                    wordcloud.generate_from_frequencies(frequencies=terms_ranks)
                    plt.figure(figsize=(20, 10))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    # don't display axis
                    plt.axis("off")
                    # remove margin
                    plt.tight_layout(pad=0)
                    plt.savefig(dir_out + "/Greece_" + week )
                except:
                    pass

