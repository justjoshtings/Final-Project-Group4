"""
kaggle_dataset_down.py
Script to download Kaggle datasets

author: @justjoshtings
created: 4/1/2022
"""

import time
from Woby_Modules.KaggleAPI import KaggleAPI

def main():
    kaggle_dataset_owner = 'justjoshtings'
    path_to_data = '../corpus/'
    data_url_end_point = 'spooky-reddit-stories'
    data_title = 'Spooky Reddit Stories'
    message = 'New update of corpus'

    kaggle = KaggleAPI(kaggle_dataset_owner)
    
    # Create/Update dataset
    kaggle.create_dataset(path_to_data=path_to_data, data_url_end_point=data_url_end_point, data_title=data_title, message=message)
    
    # Download dataset
    kaggle.download_dataset(owner=kaggle_dataset_owner, data_url_end_point=data_url_end_point, path_to_data='../corpus_data')

if __name__ == "__main__":
    print("Executing kaggle_dataset_down.py")
    main()