"""
scrape_reddit.py
Script to scrape Reddit to get corpus data and then save to disk/MongoDB

author: @justjoshtings
created: 3/27/2022
"""
import os

from MongoDBInterface import MongoDBInterface
from RedditAPI import RedditAPI
from CorpusProcessor import CorpusProcessor
from Woby_keys.reddit_keys import reddit_credentials

# REDDIT_API_LOG = './Woby_Log/RedditAPI.log'
# MONGODB_LOG = './Woby_Log/MongoDB.log'
# CORPUS_PROCESSOR_LOG = './Woby_Log/CorpusProcessor.log'
SCRAPPER_LOG = './Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = './corpus/'

def run_dev_api():
    os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')

    host = 'localhost'
    port = 27017
    database = 'woby_tales_corpus'
    collection = 'reddit_stories'

    subreddits = ['nosleep','stayawake','DarkTales','LetsNotMeet','shortscarystories','Thetruthishere','creepyencounters','truescarystories','Glitch_in_the_Matrix','Paranormal','Ghoststories']
    # subreddits = ['DarkTales',]
    sort_type = 'top'
    time_type = 'all'
    limit = 5000

    woby_db = MongoDBInterface(host, port, database, collection, log_file=SCRAPPER_LOG)
    reddit_connection = RedditAPI(reddit_credentials, connection_name='WobyBot', log_file=SCRAPPER_LOG)
    parser = CorpusProcessor(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

    for subreddit in subreddits:
        for res in reddit_connection.get_posts(subreddit=subreddit, sort_type=sort_type, time_type=time_type, limit=limit):
            parser.parse_response(res, db=woby_db, save_data=True)

    # last doc_id: 10791
        

def run_psaw():
    # https://api.pushshift.io/reddit/search/submission/?subreddit=DarkTales&metadata=true&size=0&after=1483246800
    os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')

    host = 'localhost'
    port = 27017
    database = 'woby_tales_corpus'
    collection = 'reddit_stories'

    subreddits = ['nosleep','stayawake','DarkTales','LetsNotMeet','shortscarystories','Thetruthishere','creepyencounters','truescarystories','Glitch_in_the_Matrix','Paranormal','Ghoststories']
    # subreddits = ['nosleep',]
    sort_type = 'score'
    sort = 'desc'
    limit = 20000

    woby_db = MongoDBInterface(host, port, database, collection, log_file=SCRAPPER_LOG)
    reddit_connection = RedditAPI(reddit_credentials, connection_name='WobyBot', log_file=SCRAPPER_LOG)
    parser = CorpusProcessor(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

    for subreddit in subreddits:
        for res in reddit_connection.psaw_query(subreddit=subreddit, sort_type=sort_type, sort=sort, limit=limit, is_video=False):
            parser.psaw_parse_response(res, db=woby_db, save_data=True)

if __name__ == "__main__":
    print("Executing scrape_reddit.py")
    # run_dev_api()
    run_psaw()