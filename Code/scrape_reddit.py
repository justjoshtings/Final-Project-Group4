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

REDDIT_API_LOG_FILENAME = './Woby_Log/RedditAPI.log'
MONGODB_LOG_FILENAME = './Woby_Log/MongoDB.log'
CORPUS_PROCESSOR_LOG_FILENAME = './Woby_Log/CorpusProcessor.log'
CORPUS_FILEPATH = './corpus/'

def main():
    os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')

    host = 'localhost'
    port = 27017
    database = 'woby_tales_corpus'
    collection = 'reddit_stories'

    subreddit = 'nosleep'
    sort_type = 'top'
    time_type = 'all'
    limit = 2

    woby_db = MongoDBInterface(host, port, database, collection, log_file=MONGODB_LOG_FILENAME)

    reddit_connection = RedditAPI(reddit_credentials, connection_name='WobyBot', log_file=REDDIT_API_LOG_FILENAME)
    res = reddit_connection.get_posts(subreddit = subreddit, sort_type=sort_type, time_type=time_type, limit=limit)
    
    parser = CorpusProcessor(CORPUS_FILEPATH, log_file=CORPUS_PROCESSOR_LOG_FILENAME)
    parser.parse_response(res, db=woby_db, save_data=True)

    '''
    Delete downloaded, Remove selftext from metadata
    Add corpus logs
    Add RedditAPI logs
    Complete MongoDBInterface logs
    Test
    '''

    # woby_db.insert_documents(sample_story)
    # woby_db.delete_documents(delete_query, delete_many=False, delete_all=True)


if __name__ == "__main__":
    print("Executing scrape_reddit.py")
    main()