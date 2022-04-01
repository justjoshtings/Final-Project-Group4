"""
test_RedditAPI.py
Testing of RedditAPI module

author: @justjoshtings
created: 3/27/2022
"""
import requests
from Woby_keys.reddit_keys import reddit_credentials
from Woby_Modules.Logger import MyLogger
from datetime import datetime
import os
import pandas as pd
from sys import getsizeof

# os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')
LOG_FILENAME = './Woby_Log/RedditScrapper.log'
CORPUS_FILEPATH = './corpus/'

reddit_client_id = reddit_credentials['personal_use_script']
reddit_secret_token = reddit_credentials['secret_key']
reddit_username = reddit_credentials['username']
reddit_password = reddit_credentials['password']

# note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
print("[Connecting to Reddit Dev App]...")
auth = requests.auth.HTTPBasicAuth(reddit_client_id, reddit_secret_token)

# here we pass our login method (password), username, and password
data = {'grant_type': 'password',
        'username': reddit_username,
        'password': reddit_password}

# setup our header info, which gives reddit a brief description of our app
headers = {'User-Agent': 'WobyBot/0.0.1'}

# send our request for an OAuth token
print("[Connecting to Reddit Account]...")
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

# convert response to JSON and pull access_token value
TOKEN = res.json()['access_token']
print("[Generated Token]...", TOKEN)

# add authorization to our headers dictionary
headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

# while the token is valid (~2 hours) we just add headers=headers to our requests
requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)


# make a request for the trending posts in /r/Python
res = requests.get("https://oauth.reddit.com/r/nosleep/top",
                   headers=headers, params={'limit': '2', 't': 'all'})

df = pd.DataFrame()  # initialize dataframe
count = 0

# loop through each post retrieved from GET request
for post in res.json()['data']['children']:

    # If not video (False) and is not link (True) and kind of post is 't3' (Link)
    if not post['data']['is_video'] and post['data']['is_self'] and post['kind'] == 't3':

        text = post['data']['selftext']
        full_name = post['kind']+'_'+post['data']['id']
        doc_id = str(count)
        corpus_savepath = CORPUS_FILEPATH+post['data']['subreddit']

        # append relevant data to dataframe
        df = pd.concat([df, pd.DataFrame.from_records([{
            'doc_id': doc_id,
            'full_name': full_name,
            'subreddit': post['data']['subreddit'],
            'subreddit_name_prefixed': post['data']['subreddit_name_prefixed'],
            'title': post['data']['title'],
            'little_taste': text[:100],
            'selftext': post['data']['selftext'],
            'author_fullname': post['data']['author_fullname'],
            'author': post['data']['author'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'ups': post['data']['ups'],
            'downs': post['data']['downs'],
            'score': post['data']['score'],
            'num_comments': post['data']['num_comments'],
            'permalink': post['data']['permalink'],
            'kind': post['kind'],
            'num_characters': len(text),
            'num_bytes': getsizeof(text),
            'created_utc': post['data']['created_utc'],
            'created_human_readable': datetime.utcfromtimestamp(post['data']['created_utc'],).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'filepath':corpus_savepath+f'/{doc_id}_{full_name}.txt'
        }])], ignore_index=True)

        if not os.path.exists(corpus_savepath):
            os.makedirs(corpus_savepath)

        try:
            with open(corpus_savepath+f'/{doc_id}_{full_name}.txt', 'w', encoding='utf-8') as f:
                f.write(text)
        except FileNotFoundError:
            print(
                f'[FileNotFoundError] Error saving, skipping {doc_id}_{full_name}')

        count += 1


print(df.shape, df.size)
print(df.head())