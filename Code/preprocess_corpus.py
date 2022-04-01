"""
preprocess_corpus.py
Script to preprocess corpus data

author: @justjoshtings
created: 3/31/2022
"""
import os

from Woby_Modules.MongoDBInterface import MongoDBInterface
from Woby_Modules.CorpusProcessor import CorpusProcessor

SCRAPPER_LOG = './Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = './corpus/'

os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')

host = 'localhost'
port = 27017
database = 'woby_tales_corpus'
collection = 'reddit_stories'

woby_db = MongoDBInterface(host, port, database, collection, log_file=SCRAPPER_LOG)
parser = CorpusProcessor(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

documents = woby_db.get_documents(query={}, sort=[], projection={}, limit=1000000, size=False, show=False)

size = 0
for doc in documents:
    size += doc['num_bytes']

print(int(size)/1e6, 'MBs')

n_stories = 0
for sub in os.listdir('./corpus/'):
    n_stories += len(os.listdir(f'./corpus/{sub}'))

print(n_stories)

