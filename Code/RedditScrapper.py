"""
RedditScrapper.py
Object to handle scrapping of Reddit posts from various sub-reddits.

author: @justjoshtings
created: 3/27/2022

with help from: https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
"""
import requests
from Woby_keys.reddit_keys import secret_key, personal_use_script, username, password
from Logger import MyLogger
from datetime import datetime
import os

os.chdir(os.path.expanduser("~")+'/Final-Project-Group4/')
LOG_FILENAME = './Woby_Log/RedditScrapper.log'

reddit_client_id = personal_use_script
reddit_secret_token = secret_key
reddit_username = username
reddit_password = password

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
