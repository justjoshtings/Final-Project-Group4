"""
RedditAPI.py
Object to handle scrapping of Reddit posts from various sub-reddits.

author: @justjoshtings
created: 3/27/2022

Reddit Dev API Documentations: 
        - https://www.reddit.com/dev/api
        - https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
"""
import requests
from Woby_keys.reddit_keys import reddit_credentials
from Logger import MyLogger
from datetime import datetime
import os
import pandas as pd
from sys import getsizeof
import psaw


class RedditAPI:
	'''
	Object to handle Reddit Dev API.
	'''
	def __init__(self, reddit_credentials, connection_name='MyBot', log_file=None):
		'''
		Params:
			self: instance of object
			reddit_credentials (dict): reddit_credentials = {'app_name':'', 
															'personal_use_script':'',
															'secret_key':'',
															'redirect_uri':'',
															'username':'',
															'password':''
															}
			connection_name (str): Optional string to pass to name connection, default = 'MyBot'
			log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
		'''
		self.cwd = os.path.expanduser("~")+'/Final-Project-Group4/'
		os.chdir(self.cwd)
		print(f'Changing current directory to {self.cwd}')
		
		self.reddit_client_id = reddit_credentials['personal_use_script']
		self.reddit_secret_token = reddit_credentials['secret_key']
		self.reddit_username = reddit_credentials['username']
		self.reddit_password = reddit_credentials['password']

		self.LOG_FILENAME = log_file
		if self.LOG_FILENAME:
            # Set up a specific logger with our desired output level
			self.mylogger = MyLogger(self.LOG_FILENAME)
			# global MY_LOGGER
			self.MY_LOGGER = self.mylogger.get_mylogger()
			self.MY_LOGGER.info(f"{datetime.now()} -- [FILE SETUP] f'Changing current directory to {self.cwd}...")
	
	def handshake(self):
		'''
		Method to perform connection handshake to Reddit Dev API service and obtain token valid for 2 hours.

		Params:
			self: instance of object
		'''
		# note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
		print("[Connecting to Reddit Dev App]...")
		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [Connecting to Reddit Dev App]...")
		
		self.auth = requests.auth.HTTPBasicAuth(self.reddit_client_id, self.reddit_secret_token)

		# here we pass our login method (password), username, and password
		self.login_data = {'grant_type': 'password',
					 'username': self.reddit_username,
					 'password': self.reddit_password}

		# setup our header info, which gives reddit a brief description of our app
		self.headers = {'User-Agent': 'WobyBot/0.0.1'}

		# send our request for an OAuth token
		print("[Connecting to Reddit Account]...")
		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [Connecting to Reddit Account]...")

		self.res = requests.post('https://www.reddit.com/api/v1/access_token',
							auth=self.auth, data=self.login_data, headers=self.headers)

		# convert response to JSON and pull access_token value
		self.TOKEN = self.res.json()['access_token']
		
		print("[Generated Token]...", self.TOKEN)
		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [Generated Token]...")

		# add authorization to our headers dictionary
		self.headers = {**self.headers, **{'Authorization': f"bearer {self.TOKEN}"}}

		# while the token is valid (~2 hours) we just add headers=headers to our requests
		requests.get('https://oauth.reddit.com/api/v1/me', headers=self.headers)

	
	def get_posts(self, subreddit, sort_type, time_type, limit=25):
		'''
		Method to get posts of a particular subreddit.

		Params:
			self: instance of object
			subreddit (str): subreddit to pull data from
			sort_type (str): type of sorting to rank posts, ('top', 'best', 'hot', 'new', 'random', 'rising')
			limit (int): how many posts to pull each GET, default=25, max = 100
		'''
		self.handshake()

		# To handle limit of 100 when we want to get more posts than 100
		runs_limit_100 = limit // 100
		last_run_limit = limit % 100

		if last_run_limit:
			total_calls = runs_limit_100+1
		else:
			total_calls = runs_limit_100

		self.params = {'limit': str(limit), 't': f'{time_type}'}

		for i in range(total_calls):
			if i < runs_limit_100: 
				limit = 100
			else:
				limit = last_run_limit

			self.params['limit'] = str(limit)

			# make a request for the trending posts in /r/Python
			self.res = requests.get(f"https://oauth.reddit.com/r/{subreddit}/{sort_type}",
							headers=self.headers, params=self.params)

			if self.LOG_FILENAME:
				self.MY_LOGGER.info(f"{datetime.now()} -- [Requesting] https://oauth.reddit.com/r/{subreddit}/{sort_type}...")
			
			try:
				full_name = self.res.json()['data']['children'][-1]['kind']+'_'+self.res.json()['data']['children'][-1]['data']['id']
			except IndexError:
				print(self.res.json())
				print(f"[Requesting Error] End of posts! {IndexError}")
				if self.LOG_FILENAME:
					self.MY_LOGGER.info(f"{datetime.now()} -- [Requesting Error] End of posts! {IndexError}")
				
				break
			
			yield self.res
			
			# add/update fullname in params
			self.params['after'] = full_name


	def psaw_query(self, subreddit, sort_type='created_utc', sort='desc', limit=25, metadata='false', is_video=False):
		'''
		Reddit Dev API has a submissions limit pull of 1000, we can use PushShift service to get more posts.
		https://github.com/pushshift/api

		Params:
			self: instance of object
			subreddit (str): subreddit to pull data from
			sort_type (str): sort submissions by ("score", "num_comments", "created_utc"), default = 'created_utc'
			sort (str): sort type ("asc", "desc"), default = 'desc'
			metadata (str): display metadata about the query, ["true", "false"], default = 'false'
			limit (int): how many posts to pull each GET, default=25
		'''
		api = psaw.PushshiftAPI()

		print(f"[Requesting] subreddit:{subreddit}, sort_type:{sort_type}, sort:{sort}, limit:{limit}, metadata:{metadata}, is_video:{is_video}")
		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [Requesting] subreddit:{subreddit}, sort_type:{sort_type}, sort:{sort}, limit:{limit}, metadata:{metadata}, is_video:{is_video}")
		
		for response in api.search_submissions(subreddit="DarkTales", sort_type=sort_type, sort=sort, limit=limit, is_video=is_video):
			yield response


if __name__ == "__main__":
    print("Executing RedditAPI.py")
else:
    print("Importing RedditAPI")