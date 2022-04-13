"""
clear_data.py
Script to remove all data from MongoDB and locally saved files to restart data collection

author: @justjoshtings
created: 4/11/20
"""
from Woby_Modules.MongoDBInterface import MongoDBInterface

def main():
    LOG_FILENAME = '../Woby_Log/MongoDB.log'

    host = 'localhost'
    port = 27017
    database = 'woby_tales_corpus'
    collection = 'reddit_stories'
    
    woby_db = MongoDBInterface(host, port, database, collection, log_file=LOG_FILENAME)
    
    documents = woby_db.get_documents(limit=100000, size=False, show=False)
    print('Total Number of Docs in MongoDB: ',len(documents))
    
    woby_db.delete_documents({}, delete_many=False, delete_all=True)

    documents = woby_db.get_documents(limit=100000, size=False, show=False)
    print('Total Number of Docs in MongoDB: ',len(documents))

if __name__ == "__main__":
    print("Executing clear_data.py")
    main()