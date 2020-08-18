# SPARQL queries explained
After etracting terms from the tweets using various methodes (H-TFIDF, TF-IDF, TF), we seek to know how many of them are in controlled vocabulary (i.e Thesaurus) like Agrovoc and MeSH.
## Agrovoc
**endpoint**: http://agrovoc.uniroma2.it/sparql

**query**:
```
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>

select ?myterm ?labelAgro 
WHERE {
    ?myterm skosxl:literalForm ?labelAgro.
    FILTER(lang(?labelAgro) = "en").
    filter(REGEX(?labelAgro, "^coronavirus(s)*$", 'i'))
}
```
Result of this query (but for the term "symptom" instead of "coronavirus" becaut it's in Agrovoc) on Agrovoc Website : http://agrovoc.uniroma2.it/sparql
![agrovoc](img-markdown/agrovoc.png) 
**explanation**:
We want to retrieve 2 variables that we names myterm and labelAgro :
* labelAgro is a LiteralFrom predicat which is enlish and match our term. Here, for the example, our term is "cornavirus". The regex add an optional "s" at this end of term because, in Agrovoc, concept in english are in their plural form (unlike concept in French)
* myterm is concept which as a LiteralForm predicat that match with these 2 requirements explained above (label is English and matching the regexs)
## MeSH :
**endpoint**: https://id.nlm.nih.gov/mesh/

**query**:
```
SELECT ?d ?dName ?c ?cName
FROM <http://id.nlm.nih.gov/mesh>
WHERE {
    ?d a meshv:Descriptor .
    ?d meshv:concept ?c .
    ?d rdfs:label ?dName .
    ?c rdfs:label ?cName
    FILTER(REGEX(?dName,'^coronavirus$','i'))
}
ORDER BY ?d
```
Results of this query from MeSH website : https://id.nlm.nih.gov/mesh/query
![mesh-results](img-markdown/mesh-results.png)
**explanation**:
We want to retrieve 4 variables:
* ?d: A mesh descriptor that have a "label" predicat which match with the regex
* ?dName : The mesh descriptor label
* ?c: concept of the mesh descriptor
* ?cName: Label of the previous concept