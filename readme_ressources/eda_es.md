# Here you can find other experimentations to evaluate H-TFIDF by comparing it with a classical TF-IDF :

6. Match term extracted by our various methods with thesaurus like Agrovoc or MeSH using [SPARQL](elasticsearch/analyse/sparqlquery.md)

    Compute the percentage of terms extracting by TF / TF-IDF / H-TF-IDF **included** in the 4 thesuarus : Wordnet / Agrovoc / Mesh

    ![eval 6](readme_ressources/thesaurus_coverage.png)
    ![eval 6 - 2](readme_ressources/venn_wordcloud.png)
7. For the 3 measures first 100 terms : compares occurence of terms in states aggregated tweets
8. For the K most frequent terms in state aggregated tweets : compute % of the 3 measures coverage

    The aim is to compute % of overlapping between the K most frequent words using by state (i.e : concat all tweets by state) **with** N terms extracted by TF / TF-IDF / H-TF-IDF.
    TF and TF-IDF have been parameterized with Doc = 1 tweet and Corpus = UK.

    ![eval 8](readme_ressources/eval8_update_nb-top-term.png "Evaluation of point 8")

    H-TF-IDF works better than TF/TFIDF in general and in particular with small amount of tweets (Nothern Ireland)
9. Evaluation with different TF / TF-IDF settings : work on corpus and document

    | H-TF-IDF  |  TF / TF-IDF UK | TF / TF-IDF by state |
    |:---|:---|:---|
    | Doc: All tweets aggregate by state  | Doc : 1 tweet  | Doc : 1 tweet  |
    | Corpus: UK  | Corpus: UK  | Corpus: for each state  |

    For the past evaluations, we use TF/TF-IDF with Doc = 1 tweet and Corpus = UK. During the presentation on text mining activities with the Executive Board of MOOD (2020-09-01), Bruno Martins suggest us to compare H-TF-IDF with TF / TF-IDF on corpus = state.

    Some results

    ![eval9_venn_corpus-state](readme_ressources/eval9_venn_corpus-state.png "Venn wordcloud for corpus = state")
    ![eval9_barchart_corpus-state](readme_ressources/eval9_barchart_corpus-state.png "Barchart common or specific words by measure")
    ![eval9_statecoverage_corpus-state](readme_ressources/eval9_statecoverage_corpus-state.png "Barchart state coverage")

    Few comments: When focus on state, TF and TF-IDF have better overlap with K first terms most frequent in state, except for England (which as much more tweets).

    **So what we can say is H-TFIDF work better than TF/ TF-IDF for both on small (show in eval 8) and big (eval 9) amount of tweets**
10. Evalutate the power of discrimination of H-TFIDF on space
    For each measure, count the number of word which are retrieve for only one state to show that H-TFIDF can retrieve terms used only in one place

    Some results
    ![eval10_uniquelocation_corpus-uk](readme_ressources/eval10_uniquelocation_corpus-uk.png "percentage of unique location for words extracted")
    ![eval10_uniquelocation_corpus-state](readme_ressources/eval10_uniquelocation_corpus-state.png "percentage of unique location for words extracted")

    Few comments :
    From matrix build in eval 8 and 9 (state coverage), we drop all duplicates, count word that have unique location and normalize on the length of extracted term by measure.

    **We see that H-TFIDF have more word with unique location that the others. It can then extract terms specific of a location**.

    H-TFIDF works better thant TF/TF-IDF for state with big amount of tweets (England)
11. Retrieve tweets corresponding at best ranked terms extracted by H-TF-IDF

    a. What we want to evaluate is : Is H-TF-IDF retrieve terms from tweets from the specific state ? i.e. : if the world "w" is extracted by H-TF-IDF for state "s", is the w in also other state ?
    To achieve that, we query elastic search to get all tweets (and their location) containing terms extracted from H-TF-IDF and TF-IDF (corpus = state). Then compute the ratio of Sum of count of tweet with state associated with the term extracted / sum of count of all Tweet.

    Some results :
    ![eval11_htfidf](readme_ressources/eval11_nb_tweets_ratio_specific-on-all_states_HTFIDF.png)
    ![eval11_tfidf](readme_ressources/eval11_nb_tweets_ratio_specific-on-all_states_TF-IDF.png)

    Few comments:
    These figures doesn't show the discriminance power because of terms very frequent that are common used in England. For example  the term "worried" best H-TF-IDF of Nothern Ireland is in lot of tweets from England as well.
    This over frequently term affect to much theses barchart.

    b. Compute the mean of nb of states for each term

    For each state, compute the mean of number of states that a tweet containing terms of measure have been sent.

    Some results:
    ![eval11b_htfidf](readme_ressources/eval11b_mean_nb-states_htfidf.png)
    ![eval11b_tfidf](readme_ressources/eval11b_mean_nb-states_tfidf.png)

    Few comments:

    Tweet containing terms extracted from H-TF-IDF are from less different states that TF-IDF (exept for England) that show H-TF-IDF have a better space discriminance.

    What we can also say is H-TF-IDF could be very effective when we compare different area with large disparities int the number of tweets (For example : England with Nothern Ireland). This could be very usefull for analyse European countries.

12. Create a choropleth maps on count of tweets by UK states

    ![eval12](readme_ressources/eval12_maps-Uh_with-Nb-Tweets.png)
13. Bert Summarize tweets containing terms from the TOP100 H-TF-IDF