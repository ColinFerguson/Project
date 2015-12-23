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
from pdfminer.converter import TextConverter  # XMLConverter, HTMLConverter
import urllib2
from urllib2 import Request
import datetime
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from cStringIO import StringIO
import sys
from nltk.stem import WordNetLemmatizer
import multiprocessing


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
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)

    # Create a PDF interpreter object
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Process each page contained in the document
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        data = retstr.getvalue()
    return data


def parsePDF2(url):

    '''parsePDF2 is an improvement of parsePDF because it adds waits
    to avoid getting kicked off of the site you are scraping.  Also added
    are the try/except blocks to deal with pdfs that won't open.'''

    try:
        open = urllib2.urlopen(Request(url[0])).read()

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
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)

        # Create a PDF interpreter object
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # Process each page contained in the document
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            data = retstr.getvalue()
        sl = random.randint(0, 30)
        time.sleep(sl)
        return data, url[0], url[1]

    except:
        x = random.randint(5, 15)
        time.sleep(x)
        try:
            open = urllib2.urlopen(Request(url[0])).read()

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
            device = TextConverter(rsrcmgr, retstr, codec=codec,
                                   laparams=laparams)

            # Create a PDF interpreter object
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            # Process each page contained in the document
            for page in PDFPage.create_pages(document):
                interpreter.process_page(page)
                data = retstr.getvalue()
            return data, url[0], url[1]
        except:
            time.sleep(random.randint(5, 10))


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
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password, caching=caching,
                                  check_extractable=True):
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
    r = requests.get(base_url, headers={'User-agent': 'Mozilla/5.0'})
    print r.status_code
    soup = BeautifulSoup(r.text, 'html.parser')
    pdfs = soup.findAll(title='Download PDF')
    links = [str(pdf).split()[1].strip('href="') for pdf in pdfs]
    urls = ['http://lanl.arxiv.org' + link for link in links]

    titles = soup.findAll(class_="list-title")
    title_list = []
    for ix in range(len(titles)):
        title_list.append(titles[ix].text)
    return title_list, urls, links, pdfs


def get_text2(url):
    '''Input a URL from the arxiv (page of a list of papers), return a list of
    parsed articles (list of strings).  get_text2 is better than get_text
    because it implements parallel processing.
    '''

    base_url = url
    r = requests.get(base_url, headers={'User-agent': 'Mozilla/5.0'})

    print "Status: ", r.status_code

    soup = BeautifulSoup(r.text, 'html.parser')
    pdfs = soup.findAll(title='Download PDF')
    links = [str(pdf).split()[1].strip('href="') for pdf in pdfs]
    urls = ['http://lanl.arxiv.org' + link for link in links]
    titles = soup.findAll(class_="list-title")
    title_list = []
    for ix in range(len(titles)):
        title_list.append(titles[ix].text)
    url_title = zip(urls, title_list)

    articles = []
    failed = []
    good_urls = []
    pool = multiprocessing.Pool(processes=4)
    articles = pool.map(parsePDF2, url_title)

    print "Returned {0} articles".format(len(articles))
    print "Status: ", r.status_code
    return articles


def clean_pdf_text(text):
    '''Input: list of text documents, each element of the list is a
    single (long) string. Output: slightly cleaned up version,
    try to get rid of strange characters and short 'words'
    '''
    S = set()
    S.update(letter for letter in string.lowercase)
    S.update(letter for letter in string.uppercase)
#    S.update(digit for digit in string.digits)
    new_text = []
    for i in range(len(text)):
        new_text.append([word.lower() for word in text[i].split() if
                        (word[0] in S) and (word[-1] in S) and
                         (len(word) > 3)])

    for i in range(len(new_text)):
        new_text[i] = ' '.join(new_text[i])

    return new_text


def math_stop():
    '''Add math specific words to the standard stop list'''
    tfidf = TfidfVectorizer(stop_words='english')
    Stop = set()
    Stop.update([word for word in tfidf.get_stop_words()])
    Stop.update(['theorem', 'denote', 'like', 'thank', 'lemma', 'proof',
                'sum', 'difference', 'corollary', 'hand',
                 'product', 'multiple', 'let', 'group',
                 'prime', 'log', 'limit', 'cid', 'result',
                 'main', 'conjecture', 'case', 'suppose',
                 'function', 'assume', 'follows',
                 'given', 'define', 'note', 'defined', 'class',
                 'proposition', 'function', 'set',
                 'primes', 'numbers', 'form', 'integers', 'curves',
                 'real', 'using', 'following', 'obtain', 'prove',
                 'definition', 'large', 'small', 'action', 'define',
                         'bound', 'sufficiently', 'subject', 'non',
                          'mathematics'])
    return list(Stop)


def get_topics_nmf(urls, num_topics):
    '''Input: URL containing links to each document (pdf) in the
    corpus (i.e. arxiv)  Output: the num_topics most important latent
    topics from the corpus (via NMF)
    '''
    article_info = []
    for url in urls:
        article_info.append(get_text(url))

    text = []
    for thing in article_info:
        text.extend(thing[0])
    text = clean_pdf_text(text)

    tfidf_math = TfidfVectorizer(max_features=100, stop_words=math_stop(),
                                 ngram_range=(1, 1), decode_error='ignore')
    M = tfidf_math.fit_transform(text)

    feature_names = tfidf_math.get_feature_names()
    feature_names = [WordNetLemmatizer().lemmatize(word)
                     for word in feature_names]
    nmf = NMF(n_components=num_topics)
    nmf.fit(M)
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        topics.append((" ".join([feature_names[i] for i in
                                topic.argsort()[:-10 - 1:-1]])))
    return M, topics, text, title_list, urls

def get_topics_lda(urls, num_topics):
    '''Similar to get_topics_nmf, but using LDA for topic extraction.
    This method is prefered because LDA is a probabilistic model, and thus
    you can see that probability of a given text falling into the various topics.
    This function needs to be cleaned up, right now it is only printing, not returning anything.'''

    article_info = []
    for url in urls:
        article_info.append(get_text(url))

    text = []
    for thing in article_info:
        text.extend(thing[0])
    text = clean_pdf_text(text)
    countvec = CountVectorizer(decode_error='ignore', stop_words=Stop, max_features=5000)
    CV = countvec.fit_transform(new_text)
    vocab=tuple(countvec.vocabulary_)
    lda_model = lda.LDA(n_topics=num_topics, n_iter=1500)
    lda_model.fit(CV)
    topic_word = lda_model.topic_word_  # model.components_ also works
    n_top_words = 8
    topic_words = []
    for i, topic_dist in enumerate(topic_word):
    topic_words.append(np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1])

    for i in range(len(topic_words)):  # this gives the topics
    print ' '.join(topic_words[i])
    print '\n'

    doc_topic = lda_model.doc_topic_  # This will give the most probable topic for each document
                                      # (only 10 listed here)
    topic_most_pr = []
    for n in range(10):
        topic_most_pr.append(doc_topic[n].argmax())

    for n in range(10):
        print "doc: {} topic: {}".format(n, str(topic_words[topic_most_pr[n]]))
    return



def get_best_titles(M, index, num_articles, title_list):
    '''Used for similarity modeling.  Input the tfidf model (M),
     and the desired number of articles, output the num_articles best matches to the
     given article.'''

    N = M.todense()
    Dists = np.zeros((N.shape[0], N.shape[0]))
    for ix in range(len(Dists)):
        for jx in range(len(Dists)):
            Dists[ix, jx] = cosine(N[ix], N[jx])
    distances = Dists[index]
    best_score_indices = np.argsort(distances)[1:num_articles]
    best_scores = [np.around(distances[i], 3) for i in best_score_indices]
    best_titles = [title_list[i] for i in best_score_indices]
    return best_titles, best_scores


def three_best(url, num_classes=3):
    '''Input: url of a pdf paper.
    Output: the num_classes most probable classifications for the given paper'''
    user_text = [parsePDF(url)]
    user_text = clean_pdf_text(user_text)
    vector = prediction_model.transform(user_text)
    top = np.squeeze(np.argsort(ovr.predict_proba(vector))[:, -num_classes:])
    top_probs = np.squeeze(ovr.predict_proba(vector)[:, top])

    header = "The most probable classifications for this paper are: "
    class_and_prob = []
    output = ''
    for ix in range(1, num_classes):
        class_and_prob.append(' <p> {0} with probability \
                    {1} <p>'.format(names[ovr.classes_[top][-ix]],
                                    top_probs[-ix]))
    return header+'\n'+'\n'.join(class_and_prob)


def NT_sim(url):
    '''Input: the url of a number theory paper (pdf)
    Output: the five most similar papers from the corpus'''
    user_text = [parsePDF(url)]
    user_text = clean_pdf_text(user_text)
    user_vec = similarity_model.transform(user_text)
    user_dense = user_vec.todense()

    if np.linalg.norm(user_dense) == 0.0:
        return "Sorry, this paper couldn't be parsed"
    else:
        distances = [cosine(x, user_dense) for x in N]
        best_score_indices = np.argsort(distances)[:6]
        best_scores = [np.around(distances[i], 3) for i in best_score_indices]
        best_titles = [title_list[i] for i in best_score_indices]
        best_urls = [urls[i] for i in best_score_indices]

        header = "The five most similar papers \
                        from the past year are: "
        sim = []

        for x, y, z in zip(best_titles, best_scores, best_urls):
            sim.append(x)
            sim.append('Distance: {0}'.format(y))
            sim.append(z)
        return header + '\n' + '\n'.join(sim)


def scrape(url_list, label, filename):
    '''Input: List of urls from a specific category to be scraped (i.e. from the arxiv),
    label corresponding to the category, filename to save the data under
    Output: 4-tuple of text, url, title, label, pickled into a file.  Be careful
    not to scrape too quickly and risk getting banned from a site.  Probably good to
    break url_list into chunks.'''

    for url in url_list:
        arts = get_text2(url)
        bad_type = 0
        good_arts = []
        for ix in range(len(arts)):
            if type(arts[ix]) == tuple:
                good_type += 1
                good_arts.append(arts[ix])
            else:
                bad_type += 1
        print "Good: ", good_type  #Don't really need these print statements, but I like to see how things are going
        print "Bad: ", bad_type
    for ix in range(len(good_arts)):
        good_arts[ix] += (label,)  #Add the label to be a target later
    with open(filename, 'w') as f:
        pickle.dump(good_arts, f)
