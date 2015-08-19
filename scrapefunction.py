#initial function for getting urls and scraping the arxiv

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

def get_text(url):
    base_url = url
    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    pdfs = soup.findAll(title = 'Download PDF')
    links = [str(pdf).split()[1].strip('href="') for pdf in pdfs]
    urls = ['http://arxiv.org'+ link for link in links]
    articles = []
    for url in urls:
        articles.append(parsePDF(url))
    return articles


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
