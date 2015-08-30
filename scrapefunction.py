#initial function for getting urls and scraping the arxiv
import pdfminer
import requests
from bs4 import BeautifulSoup
from pdfminer import pdfparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import string
from scipy.spatial.distance import cosine
import numpy as np
from nltk import PorterStemmer
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import lda
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import  TextConverter # , XMLConverter, HTMLConverter
import urllib2
from urllib2 import Request
import datetime
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from cStringIO import StringIO




# Define a PDF parser function
def parsePDF(url):

    # Open the url provided as an argument to the function and read the content
    open = urllib2.urlopen(Request(url)).read()

    # Cast to StringIO object
    from StringIO import StringIO
    memory_file = StringIO(open)

    # Create a PDF parser object associated with the StringIO object
    parser = PDFParser(memory_file)

    # Create a PDF document object that stores the document structure
    document = PDFDocument(parser)

    # Define parameters to the PDF device objet
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    codec = 'utf-8'

    # Create a PDF device object
    device = TextConverter(rsrcmgr, retstr, codec = codec, laparams = laparams)

    # Create a PDF interpreter object
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Process each page contained in the document
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        data =  retstr.getvalue()
    return data

def convert_pdf_to_txt(path):
    '''Input a pdf file (on local disk), output the text of the file
    '''

    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,\
                                  password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text

def get_text(url):
    '''Input a URL from the arxiv (page of a list of papers), return a list of
    parsed articles (list of strings)
    '''

    base_url = url
    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    pdfs = soup.findAll(title = 'Download PDF')
    links = [str(pdf).split()[1].strip('href="') for pdf in pdfs]
    urls = ['http://arxiv.org'+ link for link in links]
    articles = []
    for url in urls:
        articles.append(parsePDF(url))

    titles = soup.findAll(class_="list-title")
    title_list = []
    for ix in range(len(titles)):
        title_list.append(titles[ix].text)

    return articles, title_list




def clean_pdf_text(text):
    '''Input: list of text documents, each element of the list is a single (long) string
    Output: slightly cleaned up version, try to get rid of strange characters and short 'words'
    '''
    S = set()
    S.update(letter for letter in string.lowercase)
    S.update(letter for letter in string.uppercase)
    S.update(digit for digit in string.digits)

    new_text=[]
    for i in range(len(text)):
        new_text.append([word.lower() for word in text[i].split() if (word[0] in S) and (word[-1] in S)\
                    and (len(word)>3)])

    for i in range(len(new_text)):
        new_text[i] = ' '.join(new_text[i])

    return new_text


def math_stop():
    '''Add math specific words to the standard stop list'''
    tfidf = TfidfVectorizer(stop_words='english')
    Stop = set()
    Stop.update([word for word in tfidf.get_stop_words()])
    Stop.update(['theorem', 'lemma', 'proof', 'sum', 'difference', \
                 'product', 'multiple', 'let', 'group', 'prime', 'log', 'limit', 'cid', 'result'\
                'main', 'conjecture', 'case', 'suppose', 'function', 'assume', 'follows', \
                'given', 'define', 'note', 'defined', 'class', 'proposition', 'function', 'set', \
                 'primes', 'numbers','form', 'integers', 'curves', 'real'])
    return list(Stop)


def get_topics(url, num_topics):
    '''Input: URL containing links to each document (pdf) in the corpus (i.e. arxiv)
    Output: the num_topics most important topics from the corpus
    '''
    text = get_text(url)
    clean_text = clean_pdf_text(text)

    tfidf_math = TfidfVectorizer(max_features=100, stop_words=math_stop(), \
                    ngram_range=(1, 1), decode_error='ignore')
    M = tfidf_math.fit_transform(clean_text)

    feature_names = tfidf_math.get_feature_names()
    nmf = NMF(n_components=10)
    nmf.fit(M)
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        topics.append((" ".join([feature_names[i] for i in \
                topic.argsort()[:-10 - 1:-1]]))
    return M, topics


def get_best_titles(M, index, num_articles):
    N=M.todense()
    Dists=np.zeros((N.shape[0], N.shape[0]))
    for ix in range(len(Dists)):
        for jx in range(len(Dists)):
            Dists[ix, jx]=cosine(N[ix], N[jx])
    distances = Dists[index]
    best_score_indices = np.argsort(distances)[1:num_articles]
    best_scores = [np.around(distances[i],3) for i in best_score_indices]
    best_titles = [title_list[i] for i in best_score_indices]
    return best_titles, best_scores
