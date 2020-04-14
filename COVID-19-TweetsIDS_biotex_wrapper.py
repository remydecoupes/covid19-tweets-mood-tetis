

import os,glob,shutil,sys,time
from pathlib import Path

import pandas as pd

home = str(Path.home())
biotexWrapperFizePath = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotex-wrapper-fize/biotex_python/biotex')
biotexCorpusPath = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexcorpus')
biotexResultPath = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults')


class BiotexWrapper():
    """
    Wrapper to execute and returned the result from the Biotex program
    See more about Biotex at: http://tubo.lirmm.fr:8080/biotex/index.jsp
    """
    def __init__(self,
        biotex_jar_path = os.path.join(biotexWrapperFizePath,"Biotex.jar"),
        pattern_path = os.path.join(biotexWrapperFizePath,"patterns"),
        dataset_src = os.path.join(biotexWrapperFizePath,"dataSetReference") ,
        stopwords_src = os.path.join(biotexWrapperFizePath,"stopWords"),
        treetagger_src = os.path.join(home,".tree-tagger/"),
        #type_of_terms = "all",
        type_of_terms="mutli",
        language = "english",
        score = "F-TFIDF-C_M",
        #score="C_value",
        patron_number = "10"):


        self.biotexJarPath = biotex_jar_path
        self.configuration = {
            "patternsSrc": pattern_path,
            "datasetSrc": dataset_src,
            "stopwordsSrc": stopwords_src,
            "treetaggerSrc": treetagger_src,
            "typeOfTerms": type_of_terms,
            "language": language,
            "score": score,
	        "patronNumber": patron_number
        }

        self.write_conf(self.configuration)
        self.output_data=None


    def write_conf(self,confDict):
        """
        Create the configuration file to execute Biotex
        """
        f=open("configuration.txt", 'w')
        for key in confDict.keys():
            f.write("{0}={1}\n".format(key,confDict[key]))
        f.close()

    def extract_terminology(self,inputFile,nbGram="ALL"):
        """
        Execute and extract the result returned by Biotex
        """
        if isinstance(nbGram,str):
            if nbGram != "ALL":
                print("Error : Except 'ALL' value, nbGram args in extractTerminology method can't take string arg !!!\nAvailable values: 'ALL',1,2,3,4")
                return False
        if isinstance(nbGram,int):
            if nbGram > 4 or nbGram < 0:
                print("Error : nbGram value : {0} is forbidden!\nAvailable values: 'ALL',1,2,3,4 ".format(nbGram))
                return False
        #if not isinstance(nbGram,str) or not isinstance(nbGram,int):
            #print("Error: Wrong args type :{0}!\nAvailable values: 'ALL',1,2,3,4 ".format(type(nbGram)))
            #return False
        debut=time.time()
        status=os.system("java -Xms6g -Xmx10g -jar {0} {1}".format(self.biotexJarPath, inputFile))
        print("\t Biotex jar : Done in {0} sec".format(time.time()-debut))
        if status == 1 :
            print("Biotex java program has crashed !")
            return False
        if not os.path.exists("output"):
            os.makedirs("output")

        if isinstance(nbGram,int):
            output=open("output/t{0}gram.txt".format(nbGram),'r').read()
        else:
            output=open("output/ALL_gram.txt",'r').read()
        data=[]
        for line in output.split("\n"):
            parsed=line.split(";")
            if len(parsed) == 3:
                parsed[1]=int(parsed[1])
                parsed[2]=float(parsed[2])
                data.append(parsed)
        shutil.rmtree("output")
        for f in glob.glob("to_tag_*.txt"):
            os.remove(f)
        self.output_data=data
        return self.output_data

    def terminology(self, corpus):
        try:
            self.create_corpus_from_txt(corpus)
        except:
            raise Exception("Error while creating file !")
        return pd.DataFrame(self.extract_terminology("output.txt"), columns = "term in_umls rank".split())


if __name__ == '__main__':
    # import argparse
    # parser= argparse.ArgumentParser()
    # parser.add_argument("input",help="Your Biotex input filename")
    # parser.add_argument('-s',"--sizeOfGram",help="Gram size of the term you want to extract")
    # parser.add_argument('-o',"--output",help="Output filename")
    # parser.add_argument('-d',"--debug",action="store_true",help="debug activated")
    #
    # args=parser.parse_args()
    # if args.debug:
    #     print(args)


    wrap=BiotexWrapper()
    # if args.sizeOfGram:
    #     if args.sizeOfGram != 'ALL':
    #         try:
    #             sGram=int(args.sizeOfGram)
    #         except:
    #             sGram=args.sizeOfGram
    # else:sGram="ALL"

    #data=wrap.extract_terminology(args.input,nbGram=sGram)
    print("Begin biotex Wraper")

    ## Browse date
    # for file in biotexCorpusPath.glob("english-*"):
    #     print("---------------------------")
    #     print("Work on file: "+str(file))
    #     data = wrap.extract_terminology(os.path.join(biotexCorpusPath, file.name))
    #     fileOutput = str(biotexResultPath) + "/biotex-results_"+str(file.name)
    #     print("\t save file: "+fileOutput)
    #     out_= open(fileOutput, 'w')
    #     for d in data:
    #         out_.write("\t".join(map(str, d))+"\n")
    #     out_.close()

    ## For 1 date
    # filename = 'english-tweetsOf-2020-01-26'
    # data = wrap.extract_terminology(os.path.join(biotexCorpusPath, filename))
    # fileOutput = str(biotexResultPath) + "/biotex-results_"+str(filename)
    # print("\t save file: "+fileOutput)
    # out_= open(fileOutput, 'w')
    # for d in data:
    #     out_.write("\t".join(map(str, d))+"\n")
    # out_.close()

    ## Browse date
    biotexCorpusPath = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexcorpus/subdividedcorpus')
    biotexResultPath = Path('/home/rdecoupe/PycharmProjects/covid19tweets-MOOD-tetis/biotexResults/subdividedcorpus/ftfidfc-multi')
    for file in biotexCorpusPath.glob("english-*"):
        print("---------------------------")
        print("Work on file: "+str(file))
        data = wrap.extract_terminology(os.path.join(biotexCorpusPath, file.name))
        fileOutput = str(biotexResultPath) + "/biotexftfidfc-multi-results_"+str(file.name)
        print("\t save file: "+fileOutput)
        out_= open(fileOutput, 'w')
        for d in data:
            out_.write("\t".join(map(str, d))+"\n")
        out_.close()
