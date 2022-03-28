"""
test_MongoDBInterface.py
Testing of MongoDBInterface module

author: @justjoshtings
created: 3/27/2022
"""
def main():
    from pymongo import MongoClient
    from MongoDBInterface import MongoDBInterface
    import os

    os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')
    LOG_FILENAME = './Woby_Log/MongoDB.log'

    host = 'localhost'
    port = 27017
    database = 'woby_tales_corpus'
    collection = 'reddit_stories'

    sample_story = {'story_id':'subreddit1_3', 
                    'subreddit_name':'subreddit1',
                    'post_title':'Sample Stoiry all the wayyyyywayyyyywayyyyywayyyyywayyyyywayyyyywayyyyywayyyyywayyyyywayyyyywayyyyywayyyyy',
                    'url':'www.google.com',
                    'post_date':'2022-03-27',
                    'path_to_text':'~/Final-Project-Group4/corpus/subreddit1/mytext.txt',
                    'author':'justjoshtings',
                    'n_bytes':1000,
                    'n_upvotes':100,
                    'n_downvotes':90,
                    'upvote_ratio':0.867,
                    'n_comments':39,
                    'little_taste':'You are not allowed to specify both 0 and 1 values in the same object (except if one of the fields is the _id field). If you specify a field with the value 0, all other fields get the value 1, and vice versa:'
                    }

    delete_query = {'story_id':'subreddit1_0'}
    
    woby_db = MongoDBInterface(host, port, database, collection, log_file=LOG_FILENAME)
    # woby_db.insert_documents(sample_story)
    # woby_db.delete_documents(delete_query, delete_many=False, delete_all=True)
    
    query = {}
    # 1 (for ascending) or -1 (for descending)
    sort = [('_id', -1)]
    documents = woby_db.get_documents(sort=sort, limit=1, show=True)
    try:
        print(documents[0]['doc_id'])
    except IndexError:
        print(f'Empty return {documents}')

if __name__ == "__main__":
    print("Executing test_MongoDBInterface.py")
    main()



