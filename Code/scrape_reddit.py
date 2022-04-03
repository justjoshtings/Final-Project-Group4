"""
scrape_reddit.py
Script to scrape Reddit to get corpus data and then save to disk/MongoDB

author: @justjoshtings
created: 3/27/2022
"""
import os

from Woby_Modules.MongoDBInterface import MongoDBInterface
from Woby_Modules.RedditAPI import RedditAPI
from Woby_Modules.CorpusProcessor import CorpusProcessor
from Woby_keys.reddit_keys import reddit_credentials

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus/'

# os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')

host = 'localhost'
port = 27017
database = 'woby_tales_corpus'
collection = 'reddit_stories'

subreddits = ['nosleep','stayawake','DarkTales','LetsNotMeet',
                'shortscarystories','Thetruthishere','creepyencounters',
                'TrueScaryStories','Glitch_in_the_Matrix','Paranormal',
                'Ghoststories','libraryofshadows','UnresolvedMysteries','TheChills']

def run_dev_api(host, port, database, collection, subreddits):
    sort_type = 'top'
    time_type = 'all'
    limit = 1

    woby_db = MongoDBInterface(host, port, database, collection, log_file=SCRAPPER_LOG)
    reddit_connection = RedditAPI(reddit_credentials, connection_name='WobyBot', log_file=SCRAPPER_LOG)
    parser = CorpusProcessor(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

    for subreddit in subreddits:
        for res in reddit_connection.get_posts(subreddit=subreddit, sort_type=sort_type, time_type=time_type, limit=limit):
            parser.parse_response(res, db=woby_db, save_data=False)

def run_psaw(host, port, database, collection, subreddits):
    # https://api.pushshift.io/reddit/search/submission/?subreddit=DarkTales&metadata=true&size=0&after=1483246800
    sort_type = 'score'
    sort = 'desc'
    limit = 1

    woby_db = MongoDBInterface(host, port, database, collection, log_file=SCRAPPER_LOG)
    reddit_connection = RedditAPI(reddit_credentials, connection_name='WobyBot', log_file=SCRAPPER_LOG)
    parser = CorpusProcessor(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

    for subreddit in subreddits:
        for res in reddit_connection.psaw_query(subreddit=subreddit, sort_type=sort_type, sort=sort, limit=limit, is_video=False):
            parser.psaw_parse_response(res, db=woby_db, save_data=False)

if __name__ == "__main__":
    print("Executing scrape_reddit.py")
    run_dev_api(host, port, database, collection, subreddits)
    run_psaw(host, port, database, collection, subreddits)