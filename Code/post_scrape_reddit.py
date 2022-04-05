"""
post_scrape_reddit.py
Script to run post scrape_reddit.py to:
    1. Remove duplicated stories
    2. Create .csv of metadata to save in /corpus

author: @justjoshtings
created: 3/27/2022
"""
from pymongo import MongoClient
from Woby_Modules.MongoDBInterface import MongoDBInterface
import os
import pandas as pd

def main():
    LOG_FILENAME = '../Woby_Log/MongoDB.log'

    host = 'localhost'
    port = 27017
    database = 'woby_tales_corpus'
    collection = 'reddit_stories'
    
    woby_db = MongoDBInterface(host, port, database, collection, log_file=LOG_FILENAME)
    # woby_db.delete_documents({}, delete_many=False, delete_all=True)
    
    query = {}
    projection = {'selftext':0}

    # 1 (for ascending) or -1 (for descending)
    # sort = [('_id', -1)]
    sort = []
    
    documents = woby_db.get_documents(sort=sort, projection=projection, limit=100000, size=False, show=False)
    print('Total Number of Docs in MongoDB: ',len(documents))

    def find_and_delete_dups():
        '''
        Find and delete duplicates from MongoDB metadata and corpus .txt files
        '''
        pipeline = [{"$group":{"_id":"$full_name", "count":{"$sum":1}}}, {"$match": {"_id" :{ "$ne" : 'null' } , "count" : {"$gt": 1} } },{"$project": {"full_name" : "$_id", "_id" : 0} }]
        duplicates = woby_db.find_duplciates(pipeline)

        count = 0 
        rest_doc = []
        for dup in duplicates:
            # print(dup['full_name'])
            documents = woby_db.get_documents(query=dup, sort=sort, projection=projection, limit=100000, size=False, show=False)
            
            # Delete dups
            for i in range (len(documents)):
                if i == 0:
                    first_doc = documents[0]['doc_id']
                else:
                    rest_doc.append(documents[i]['doc_id'])
                    # print(f"../corpus/{documents[i]['subreddit']}/{str(documents[i]['doc_id'])}_{documents[i]['full_name']}.txt")
                    # with open(f"../corpus/{documents[i]['subreddit']}/{str(documents[i]['doc_id'])}_{documents[i]['full_name']}.txt",'r') as f:
                    #     data = f.read()
                    # print(data)

                    os.remove(f"../corpus/{documents[i]['subreddit']}/{str(documents[i]['doc_id'])}_{documents[i]['full_name']}.txt")
                    woby_db.delete_documents({"doc_id":documents[i]['doc_id']}, delete_many=False, delete_all=False)

            count += len(documents)
        
        print('Number of documents that are duplicates: ',count)
        print('Number of duplicates minus the originals: ',len(rest_doc))

    def n_stories_saved_csv():
        '''
        Show the number of stories saved as .txt in ./corpus/
        '''
        n_stories = 0
        for sub in os.listdir('../corpus/'):
            try:
                n_stories += len(os.listdir(f'../corpus/{sub}'))
                print(sub, len(os.listdir(f'../corpus/{sub}')))
            except NotADirectoryError:
                pass

        print('N stories saved as csv:', n_stories)

    def create_metadata():
        '''
        Create metadata file from MongoDB saved data and save as .csv to ./corpus/corpus_metadata.csv
        '''
        projection = {'selftext':0}
        documents = woby_db.get_documents(sort=sort, projection=projection, limit=100000, size=False, show=False)

        corpus_metadata_df = pd.DataFrame(documents)
        corpus_metadata_df = corpus_metadata_df.sample(frac=1, axis=0, random_state=42).reset_index(drop=True)
        
        num_entries = corpus_metadata_df.shape[0]
        train_index = int(num_entries*0.8)

        corpus_metadata_df.loc[:train_index, 'train_test'] = 'train'
        corpus_metadata_df.loc[train_index:, 'train_test'] = 'test'

        # print(corpus_metadata_df.head())
        # print(corpus_metadata_df.tail())

        corpus_metadata_path = '../corpus/corpus_metadata.csv'

        corpus_metadata_df.to_csv(corpus_metadata_path, index=False)
        print(f'\n\nSaved corpus_metadata_df to {corpus_metadata_path}')


    find_and_delete_dups()
    n_stories_saved_csv()
    create_metadata()

if __name__ == "__main__":
    print("Executing post_scrape_reddit.py")
    main()



