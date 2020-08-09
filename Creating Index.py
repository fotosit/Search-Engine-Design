import urllib.request
import re
import requests
import codecs
import queue
import math
import string
import mimetypes
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from urllib.parse import unquote
from bs4 import BeautifulSoup
from urllib.parse import urlparse
# #-------------------------------------------SECTION-1 CREATING INDEX-------------------------------------------#

urlGraphList={}                 # directed url links
mediaLink=[]                    # image, video links
brokenLink=[]                   # broken links
crawledURL=[]                   # crawled links
crawlList=[]                    # frontier links
hitSites={}                     # multiHit links
index={}                        # indexed words
stemmedIndex={}                 # stemmed indexed words
stemmedIDFValueOfWords={}       # idf values of indexed words with stemmed
idfValueOfWords={}              # idf values of indexed words without stemmed
n=1000                          # number of websites to crawl
ps = PorterStemmer()            # porter stemmer

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


## Check URL is Media

def isURLMedia(url):

    mimetype,encoding = mimetypes.guess_type(url)

    isMedia = (mimetype and mimetype.startswith('image')) or (mimetype and mimetype.startswith('video')) or (
                mimetype and mimetype.startswith('audio')) or (mimetype and mimetype.startswith('application'))

    return isMedia


## Remove Header Tags

def removeHeaderTags(content):

    liste = []
    i = 0
    soup = BeautifulSoup(content, "lxml")
    tags = ["script", "style", "!--[if lt IE 7]", "meta", "class", "link", "li"]

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

## Adding the Words in the Web Page to the Index List

def contentIndexing(index, url,content):

    words=normalization(content,False)
    stemmedWords = normalization(content,True)

    for word in words:
        urlIndexing(index,word,url)

    for word in stemmedWords:
        urlIndexing(stemmedIndex,word,url)

## Web Page Indexing Process

def urlIndexing(index, keyword, url):

    if keyword in index:
        if not url in index[keyword]:
            index[keyword].append(url)

    else:
        index[keyword]=[url]


## Check URL is MultiHit

def savehitSites(url):
    if url:
        if url in hitSites:
            hitSites[url] += 1
        else:
            hitSites[url] = 1


 ## Normalization Processes

def normalization(inputString,isStem):

    sentenceArray = inputString.split()

    lowerContents = ""
    for words in sentenceArray:
        lowerContents = lowerContents + " " + words.lower()


    removePunctuationContent = lowerContents.translate(str.maketrans('', '', string.punctuation))
    removeDashtoContent = removePunctuationContent.replace("–"," ")


    withoutNumbersContent = ''.join([i for i in removeDashtoContent if
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


## Calculate IDF Values of Words

def calculateIDF(isStem):

    if isStem:
        for word in stemmedIndex:

            stemmedIDFValueOfWords.update({word: math.log(1 + (len(crawledURL) / len(stemmedIndex[word])))})
    else:
        for word in index:

            idfValueOfWords.update({word: math.log(1 + (len(crawledURL) / len(index[word])))})

## Finding All Links in the Website and Adding Them to the List

def findLinksInContents(page, url):

    outlinks=[]

    soup = BeautifulSoup(page,"html.parser")

    for div in soup.find_all("a",{'class': 'external text'}):
        div.decompose()

    for div in soup.find_all("a",{'class': 'interlanguage-link-target'}):
        div.decompose()

    tags = ["script", "style", "!--[if lt IE 7]", "meta", "class"]

    for tag in tags:
        [x.extract() for x in soup.find_all(tag)]

    for link in soup.findAll('a'):

        parsed_uri = urlparse(url)
        result = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)

        if "http" not in str(link.get("href")):
            outlinks.append(result + str(link.get("href")))
        else:
            outlinks.append(str(link.get("href")))

    return outlinks


## Adding the Found Links to the Frontier List

def addCrawlList(frontierList,outlinks):

    for url in outlinks:
        if url not in frontierList:
            frontierList.append(url)

    return frontierList

## Print Statistics

def showStatistics():

    f = open("11-Summary Statistics.txt","w")

    print("\nFrontier Url Count: %i" % (len(crawlList)))
    print("Crawled Url Count: %i" % (len(crawledURL)))
    print("Media URL Count: %i" % (len(mediaLink)))
    print("Broken Link Count: %i" % (len(brokenLink)))
    print("MultiHit URL Count: %i" % (len(hitSites)))
    print("Indexed Word Count: %i" % (len(index.keys())))
    print("Indexed Stemmed Word Count: %i" % (len(stemmedIndex.keys())))

    # Writing Summary Statistics to File
    print("\nFrontier Url Count: %i" % (len(crawlList)),file=f)
    print("Crawled Url Count: %i" % (len(crawledURL)),file=f)
    print("Media URL Count: %i" % (len(mediaLink)),file=f)
    print("Broken Link Count: %i" % (len(brokenLink)),file=f)
    print("MultiHit URL Count: %i" % (len(hitSites)),file=f)
    print("Indexed Word Count: %i" % (len(index.keys())),file=f)
    print("Indexed Stemmed Word Count: %i" % (len(stemmedIndex.keys())),file=f)

    f.close()


## Writing Statistics to Text File

def writeStatisticToText():

    writeToTxt("1-Frontier URL List",crawlList,"list")
    writeToTxt("2-Crawled Url List",crawledURL,"list")
    writeToTxt("3-Media URL List",mediaLink,"list")
    writeToTxt("4-Broken Link List",brokenLink,"list")
    writeToTxt("5-MultiHit URL List",hitSites,"dict")
    writeToTxt("6-Index List",index,"index")
    writeToTxt("7-Indexed Stemmed Word List",stemmedIndex,"index")
    writeToTxt("8-Graph List",urlGraphList,"dict")
    writeToTxt("9-IDF Values",idfValueOfWords,"idf")
    writeToTxt("10-Stemmed Word IDF Values",stemmedIDFValueOfWords,"stemmedIDF")

## Printing to a Text File

def writeToTxt(fileName, file, type):

    txtFileName = fileName + ".txt"

    with open(txtFileName,'w') as f:

        if type == "list":
            for item in file:
                f.write("%s\n" % item)

        elif type == "dict":
            for k,v in file.items():
                f.write(str(k) + " " + str(v) + '\n')

        elif type == "index":
            indexedKey = ""
            for word in file.keys():
                indexedKey = word
                for url in file[word]:
                    indexedKey += " " + url

                f.write(indexedKey + "\n")

        elif type == "idf":
            for word in file.keys():
                f.write(word + " " + str(idfValueOfWords[word]) + "\n")

        elif type == "stemmedIDF":
            for word in file.keys():
                f.write(word + " " + str(stemmedIDFValueOfWords[word]) + "\n")

    f.close()


# Crawl Process

def crawling(seed):

    url = unquote(seed)

    crawlList.append(url)

    while crawlList:

        page = crawlList.pop(0).replace("\\", "")

        decodedURL = unquote(page)

        if not isURLMedia(decodedURL):

            if decodedURL not in crawledURL:

                content,status=getPageContent(decodedURL)

                if status:
                    crawledURL.append(decodedURL)
                    beautyContent = removeHeaderTags(content)
                    contentIndexing(index,decodedURL,beautyContent)
                    outlinks = findLinksInContents(content,page)
                    addCrawlList(crawlList,outlinks)
                    urlGraphList[decodedURL] = outlinks

                    if len(crawledURL) >= n:
                        print(str(len(crawledURL)) + " sites crawled")
                        calculateIDF(False)
                        calculateIDF(True)
                        showStatistics()
                        writeStatisticToText()
                        break
                    else:
                        print(str(len(crawledURL)) + " sites crawled")

                else:
                    brokenLink.append(decodedURL)
            else:
                savehitSites(decodedURL)
        else:
            mediaLink.append(decodedURL)

    return index


crawling("https://en.wikipedia.org/")

