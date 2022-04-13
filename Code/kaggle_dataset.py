"""
kaggle_dataset.py
Script to upload/download Kaggle datasets

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
    
    print('Sleeping for 5 minutes to allow kaggle dataset to fully upload before downloading...')
    total_seconds_slept = 0
    while True:
        time.sleep(30)
        total_seconds_slept += 30
        print(f'Sleep for 30, seconds elapsed {total_seconds_slept}')
        if total_seconds_slept >= 300:
            break
    
    # Check dataset status
    kaggle.check_dataset_status(owner=kaggle_dataset_owner, data_url_end_point=data_url_end_point)

    # Download dataset
    kaggle.download_dataset(owner=kaggle_dataset_owner, data_url_end_point=data_url_end_point, path_to_data='../corpus_data')

if __name__ == "__main__":
    print("Executing kaggle_dataset.py")
    main()