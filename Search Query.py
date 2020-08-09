import urllib.request
from urllib.parse import unquote
from bs4 import BeautifulSoup
import requests
import math
from collections import OrderedDict
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import os.path
from tabulate import tabulate

#-------------------------------------------SECTION-2 SEARCH QUERY-------------------------------------------#

queryRelatedURL=[]              # The list of url associated with the query is stored
sortedResults={}                # Array where query results are stored in order
isStemmer=False                 # Stem status variable
importedHitSites={}             # the variable where the imported hit links are stored
importedIndexIDFscores={}       # The variable where the ID values of the imported index words are stored
importedIndexFile={}            # the variable where the imported index words are stored
indexedWords=[]                 # List where indexed words are stored in list format

## Import Stopped Words

f = open('SmartStopList.txt', 'r', encoding='utf-8-sig')
SmartStopList = f.read()
SmartStopKeyArray = SmartStopList.split()

## Get the Content of the Seed Page

def getPageContent(url):

    try:
         seedRequest = urllib.request.urlopen(url)
         rawSeed = seedRequest.read()
         page = rawSeed.decode("utf8")
         return page,True

    except:

        unquoteURL = urllib.parse.unquote_plus(url)

        try:
            request = requests.get(unquoteURL)

            if request.status_code == 200:

                try:
                    seedRequest = urllib.request.urlopen(unquoteURL)
                    rawSeed = seedRequest.read()
                    page = rawSeed.decode("utf8")
                    return page,True
                except:
                    return unquoteURL,False

            else:

                return unquoteURL,False

        except:
            return unquoteURL,False


# Remove Header Tags

def removeHeaderTags(content):

    liste = []
    i = 0
    soup = BeautifulSoup(content, "lxml")
    tags = ["script", "style", "!--[if lt IE 7]", "meta", "class"]

    for tag in tags:
        [x.extract() for x in soup.find_all(tag)]
    text_list = []

    for e in soup.get_text().split("\n"):
        if e != '':
            text_list.append(e)

    while i < len(text_list):
        for e in text_list[i].split():
            if len(e) > 8:
                pass
            else:
                liste.append(e)
        i += 1

    text = " ".join(liste)
    return text


 ## Normalization Processes

def normalization(inputString,isStem):

    sentenceArray = inputString.split()

    lowerContents = ""
    for words in sentenceArray:
        lowerContents = lowerContents + " " + words.lower()

    import string
    removePunctuationContent = lowerContents.translate(str.maketrans('', '', string.punctuation))

    withoutNumbersContent = ''.join([i for i in removePunctuationContent if
                                     not i.isdigit()])


    withoutStoppingWordsContent = ""
    wordsArray = withoutNumbersContent.split()

    for word in wordsArray:
        if word not in SmartStopKeyArray:
            if isStem:
                ps = PorterStemmer()
                withoutStoppingWordsContent = withoutStoppingWordsContent + " " + ps.stem(word)

            else:
                withoutStoppingWordsContent = withoutStoppingWordsContent + " " + word

    return withoutStoppingWordsContent.split()

## Importing Data From Text File

def importTxtFiles(fileName):

    f = open(fileName + '.txt','r',encoding='utf-8-sig',errors='ignore')

    if (fileName=="6-Index List") or (fileName=="7-Indexed Stemmed Word List"):
        for line in f:
            word = line.split()
            indexedWords.append(word[0])
            if len(word) > 0:
                importedIndexFile[word[0].lower()] = word[1:len(word)]

    elif (fileName=="9-IDF Values") or (fileName=="10-Stemmed Word IDF Values"):
        for line in f:
            word = line.split()
            importedIndexIDFscores[word[0].lower()] = word[1]

    elif fileName == "5-MultiHit URL List":
        for line in f:
            word = line.split()
            importedHitSites[word[0].lower()] = word[1]


## Creating Frequency Matrix

def createFrequencyMatrix(inputNormalizationList):

    frequencyList = dict.fromkeys(indexedWords, 0)

    for word in inputNormalizationList:

        if word in frequencyList:
            frequencyList[word] += 1

    return frequencyList


## Calculation of TF Values of Documents in Matrix Form

def calculateTFMatrix(inputFrequencyList, inputDocument):

    tfDict = {}
    inputDocumentCount = len(inputDocument)

    for word, count in inputFrequencyList.items():

        tfDict[word] = count / float(inputDocumentCount)

    return tfDict


## Calculation of TF Values of Documents in Matrix Form

def calculateTFIDFMatrix(tf, idfs):

    tfidf = {}

    for word, val in tf.items():

        tfidf[word] = val * float(idfs[word][0])

    return tfidf


## Calculation of TFIDF Values of Documents

def calculateDocumentTFIDF(documentURL):

    content,status = getPageContent(documentURL)

    if status:

        beautyContent = removeHeaderTags(content)
        words = normalization(beautyContent,isStemmer)
        documentFrequencyContent = createFrequencyMatrix(words)
        documentQueryTF = calculateTFMatrix(documentFrequencyContent,words)
        documentQueryTFIDF = calculateTFIDFMatrix(documentQueryTF,importedIndexIDFscores)

        return documentQueryTFIDF

    else:
        print("Content Download Failed")


## Cosine Similarity

def CalculateCosinusSimilarity(queryTFIDF, documentTFIDF):

    queryList=[]
    documentList=[]

    for key1 in queryTFIDF:
        queryList.append(queryTFIDF[key1])

    for key2 in documentTFIDF:
        documentList.append(documentTFIDF[key2])


    queryArray=np.array(queryList)
    documentArray=np.array(documentList)

    dot_product=np.dot(queryArray,documentArray)
    norm_a=np.linalg.norm(queryArray)
    norm_b=np.linalg.norm(documentArray)

    return dot_product/(norm_a*norm_b)


## Get Related Documents for Query

def executeQuery(inputQuery):

    results={}

    normalizationQuery = normalization(inputQuery,isStemmer)

    frequencyContent=createFrequencyMatrix(normalizationQuery)
    queryTF=calculateTFMatrix(frequencyContent, normalizationQuery)
    queryTFIDF = calculateTFIDFMatrix(queryTF,importedIndexIDFscores)


    for word in normalizationQuery:

        try:
            queryRelatedURL = importedIndexFile[word].split()
        except:
            queryRelatedURL = importedIndexFile[word]

    print("\nRunning Query...", flush=True)

    for url in queryRelatedURL:

        if url in importedHitSites:

            results[url]=CalculateCosinusSimilarity(queryTFIDF,calculateDocumentTFIDF(url)) + float(importedHitSites[url])*0.0001

        else:
            results[url]=CalculateCosinusSimilarity(queryTFIDF,calculateDocumentTFIDF(url))


    sortedResults= OrderedDict(sorted(results.items(),key=lambda x: x[1], reverse=True))

    if isStemmer:
        print("\nRelated links by stemmed for '%s' query: \n" % inputQuery)
    else:
        print("\nRelated links for '%s' query: \n" % inputQuery)


    writeResultsConsoleAndFile(sortedResults,inputQuery)


## Writing Query Results to a Text File

def writeResultsConsoleAndFile(inputResults, queryWords):

    result = {y:x for x,y in inputResults.items()}

    df = pd.DataFrame(result.items(), columns=['Relevance Ratio', 'URL'])
    print(df)

    if isStemmer:
        if os.path.exists("13-Stemmed Query Results.txt"):
            f = open("13-Stemmed Query Results.txt","a")
        else:
            f = open("13-Stemmed Query Results.txt","w")


        f.write("\n\nRelated links by stemmed for '%s' query:\n\n" % queryWords)
        f.write(tabulate(df, showindex=True, headers=df.columns,colalign=("left",)))
        f.write("\n\n")

    else:
        if os.path.exists("12-Query Results.txt"):
            f = open("12-Query Results.txt","a")
        else:
            f = open("12-Query Results.txt","w")

        f.write("\n\nRelated links for '%s' query:\n\n" % queryWords)
        f.write(tabulate(df, showindex=True, headers=df.columns,colalign=("left",)))
        f.write("\n\n")

    f.close()


#--------------------------------Without Stem ---------------------------------#

isStemmer=False
importTxtFiles("6-Index List")
importTxtFiles("5-MultiHit URL List")
importTxtFiles("9-IDF Values")

f=open('Query.txt',"r",encoding='utf-8-sig',errors='ignore')

for line in f.read().split(','):
    query = line.lower()
    executeQuery(query)

f.close()

indexedWords.clear()
importedIndexFile.clear()
importedIndexIDFscores.clear()

#--------------------------------With Stem ---------------------------------#

isStemmer=True  # Gövdeleme Var
importTxtFiles("7-Indexed Stemmed Word List")
importTxtFiles("10-Stemmed Word IDF Values")

f=open('Query.txt',"r",encoding='utf-8-sig',errors='ignore')

for line in f.read().split(','):
    query = line.lower()
    executeQuery(query)

print("\nQuery Complete\n")