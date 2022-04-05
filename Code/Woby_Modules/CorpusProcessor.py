"""
CorpusProcessor.py
Object to handle processing and I/O of downloaded corpus to disk.

author: @justjoshtings
created: 3/27/2022
"""
from Woby_Modules.Logger import MyLogger
from datetime import datetime
import os
import pandas as pd
from sys import getsizeof
import re
import matplotlib.pyplot as plt
import numpy as np

class CorpusProcessor:
	'''
	Object to handle processing and I/O of downloaded corpus to disk.
	'''
	def __init__(self, corpus_filepath, log_file=None):
		'''
		Params:
			self: instance of object
			log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
			corpus_filepath (str): corpus filepath to save text data to, './corpus/'
		'''
		self.corpus_filepath = corpus_filepath

		self.LOG_FILENAME = log_file
		if self.LOG_FILENAME:
            # Set up a specific logger with our desired output level
			self.mylogger = MyLogger(self.LOG_FILENAME)
			# global MY_LOGGER
			self.MY_LOGGER = self.mylogger.get_mylogger()
			self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] Setting corpus filepath to {self.corpus_filepath}...")
	
	def parse_response(self, res, db=None, save_data=False):
		'''
		Method to save posts to disk and push metadata to MongoDB

		Params:
			self: instance of object
			res (REQUEST response object): request response object to parse
			db (MongoDB database object): default None
			save_data (Boolean): default save_data = False to not save corpus to disk nor save metada to MongoDB, True to save both
		'''
		# Get last_id_count from connected DB
		if not db:
			self.last_id_count = 1
		else:
			sort = [('_id', -1)]
			documents = db.get_documents(sort=sort, limit=1, show=False)
			# If empty, that means its the first entry so set to 1
			try:
				self.last_id_count = documents[0]['doc_id'] + 1
			except IndexError:
				self.last_id_count = 1

		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] Last id count {self.last_id_count}...")
		
		df = pd.DataFrame()  # initialize dataframe
			
		# loop through each post retrieved from GET request
		for post in res.json()['data']['children']:

			# If not video (False) and is not link (True) and kind of post is 't3' (Link)
			if not post['data']['is_video'] and post['data']['is_self'] and post['kind'] == 't3':

				text = post['data']['selftext']
				full_name = post['kind']+'_'+post['data']['id']
				doc_id = self.last_id_count
				corpus_savepath = self.corpus_filepath+post['data']['subreddit']

				data_dict = {
					'doc_id': doc_id,
					'full_name': full_name,
					'subreddit': post['data']['subreddit'],
					'subreddit_name_prefixed': post['data']['subreddit_name_prefixed'],
					'title': post['data']['title'],
					'little_taste': text[:150],
					'selftext': post['data']['selftext'],
					# 'author_fullname': post['data']['author_fullname'],
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
					'filepath':corpus_savepath+f'/{doc_id}_{full_name}.txt',
					'train_test':''
				}
				
				# append relevant data to dataframe
				df = pd.concat([df, pd.DataFrame.from_records([data_dict])], ignore_index=True)

				if save_data:
					if not os.path.exists(corpus_savepath):
						os.makedirs(corpus_savepath)

					try:
						with open(corpus_savepath+f'/{doc_id}_{full_name}.txt', 'w', encoding='utf-8') as f:
							f.write(text)
						if self.LOG_FILENAME:
							self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] Saved data to {doc_id}_{full_name}...")
					except FileNotFoundError:
						print(f'[FileNotFoundError] Error saving, skipping {doc_id}_{full_name}')
						if self.LOG_FILENAME:
							self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] [FileNotFoundError] Error saving, skipping {doc_id}_{full_name}...")

					db.insert_documents(data_dict)

				self.last_id_count += 1
		
		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] Parsed json response...")
	
	def psaw_parse_response(self, res, db=None, save_data=False):
		'''
		Method to save posts to disk and push metadata to MongoDB from the PSAW (pushshift) response

		Params:
			self: instance of object
			res (REQUEST response object): request response object to parse
			db (MongoDB database object): default None
			save_data (Boolean): default save_data = False to not save corpus to disk nor save metada to MongoDB, True to save both
		'''
		# Get last_id_count from connected DB
		if not db:
			self.last_id_count = 1
		else:
			sort = [('_id', -1)]
			documents = db.get_documents(sort=sort, limit=1, show=False)
			# If empty, that means its the first entry so set to 1
			try:
				self.last_id_count = documents[0]['doc_id'] + 1
			except IndexError:
				self.last_id_count = 1

		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] Last id count {self.last_id_count}...")
		
		df = pd.DataFrame()  # initialize dataframe

		post = res[-1]

		# If not video (False) and is not link (True)
		if not post['is_video'] and post['is_self']:

			text = post['selftext']
			full_name = 't3_'+post['id']
			doc_id = self.last_id_count
			corpus_savepath = self.corpus_filepath+post['subreddit']
			try:
				upvote_ratio = post['upvote_ratio']
			except KeyError:
				upvote_ratio = -1

			data_dict = {
				'doc_id': doc_id,
				'full_name': full_name,
				'subreddit': post['subreddit'],
				# 'subreddit_name_prefixed': post['subreddit_name_prefixed'],
				'title': post['title'],
				'little_taste': text[:150],
				'selftext': post['selftext'],
				# 'author_fullname': post['author_fullname'],
				'author': post['author'],
				'upvote_ratio': upvote_ratio,
				# 'ups': post['ups'],
				# 'downs': post['downs'],
				'score': post['score'],
				'num_comments': post['num_comments'],
				'permalink': post['permalink'],
				# 'kind': post['kind'],
				'num_characters': len(text),
				'num_bytes': getsizeof(text),
				'created_utc': post['created_utc'],
				'created_human_readable': datetime.utcfromtimestamp(post['created_utc'],).strftime('%Y-%m-%dT%H:%M:%SZ'),
				'filepath':corpus_savepath+f'/{doc_id}_{full_name}.txt',
				'train_test':'',
				'api_type':'psaw'
			}

			if save_data:
				if not os.path.exists(corpus_savepath):
					os.makedirs(corpus_savepath)

				try:
					with open(corpus_savepath+f'/{doc_id}_{full_name}.txt', 'w', encoding='utf-8') as f:
						f.write(text)
					if self.LOG_FILENAME:
						self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] Saved data to {doc_id}_{full_name}...")
				except FileNotFoundError:
					print(f'[FileNotFoundError] Error saving, skipping {doc_id}_{full_name}')
					if self.LOG_FILENAME:
						self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] [FileNotFoundError] Error saving, skipping {doc_id}_{full_name}...")

				db.insert_documents(data_dict)

		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [CorpusProcessor] Parsed json response...")

	def corpus_size(self):
		'''
		Method to get corpus size in bytes

		Params:
			self: instance of object

		Return:
			total_size (float): size of folder data in bytes
        '''
		total_size = 0
		for dirpath, dirnames, filenames in os.walk(self.corpus_filepath):
			for f in filenames:
				fp = os.path.join(dirpath, f)
				# skip if it is symbolic link
				if not os.path.islink(fp):
					total_size += os.path.getsize(fp)

		return total_size

	def clean_corpus(self):
		'''
		Method to clean corpus

		Params:
			self: instance of object
		
		1. Remove all text after: "TLDR", "TLDR:", "TL;DR:", "TL DR:", "TL DR".
		2. Remove any links.
		3. Remove '&amp', '&amp;#x200B;'.
		4. Remove '***' or more.
		'''
		# Remove TLDRs & '***'& '&amp', '&amp;#x200B;' & urls
		for dirpath, dirnames, filenames in os.walk(self.corpus_filepath):
			for file in filenames:
				if file.endswith('.txt'):
					with open(dirpath+'/'+file,'r+') as f:
						text = f.read()
					
					sub_pattern = ' '
					
					pattern = rf'(TLDR|TLDR:|TL;DR:|TL;DR|TL DR:|TL DR).*'
					text = self.regex_sub(pattern, sub_pattern, text)
					self.regex_index(pattern, text)

					pattern2 = rf'\*\*\*+'
					text = self.regex_sub(pattern2, sub_pattern, text)
					self.regex_index(pattern2, text)

					pattern3 = rf'(https?://[^\s]+)'
					text = self.regex_sub(pattern3, sub_pattern, text)
					self.regex_index(pattern3, text)

					pattern4 = rf'&amp[.;#a-zA-Z0-9;]*\b'
					text = self.regex_sub(pattern4, sub_pattern, text)
					self.regex_index(pattern4, text)
					
					with open(dirpath+'/'+file,'w') as f:
						f.write(text)

	def EDA(self):
		'''
		Method to do some EDA

		Params:
			self: instance of object

		1. stories per subreddit
        2. mean number of words per sub
    	3. most common words total + by each subreddit
        4. LDA
			- network plot
			- topics plot
		'''
		results_path = './results/'

		if not os.path.exists(results_path):
			os.mkdir(results_path)

		# Number of stories per subreddit & number of words per subreddit
		sub_n_stories = dict()
		sub_n_words = dict()
		for dirpath, dirnames, filenames in os.walk(self.corpus_filepath):
			sub_n_stories[dirpath.split('/')[-1]] = len(filenames)
			n_words = 0
			for file in filenames:
				with open(dirpath+'/'+file, 'r') as f:
					text = f.read()
				words = text.split(' ')
				n_words += len(words)
			sub_n_words[dirpath.split('/')[-1]] = n_words/len(filenames)
				
		sub_n_stories.pop('')
		sub_n_words.pop('')

		subs = np.array(list(sub_n_stories.keys()))
		n_stories = np.array(list(sub_n_stories.values()))
		avg_n_words = np.array(list(sub_n_words.values()))
		
		# Number of stories per subreddit
		plt.rcParams["figure.figsize"] = (18,6)
		plt.bar(subs, n_stories, color='cadetblue')
		plt.title('Number of Stories per SubReddit', fontsize=20)
		plt.xlabel('SubReddit', fontsize=8)
		plt.ylabel('Number of Stories', fontsize=8)
		plt.xticks(rotation='70', fontsize=10)
		plt.savefig(results_path+'n_stories_per_sub.png', bbox_inches='tight')

		# Avg number of words per subreddit
		plt.rcParams["figure.figsize"] = (18,6)
		plt.bar(subs, avg_n_words, color='cadetblue')
		plt.title('Mean Number of Words per SubReddit', fontsize=20)
		plt.xlabel('SubReddit', fontsize=8)
		plt.ylabel('Mean Number of Words', fontsize=8)
		plt.xticks(rotation='70', fontsize=10)
		plt.savefig(results_path+'mean_words_per_sub.png', bbox_inches='tight')



	def regex_sub(self, pattern, sub_pattern=' ', text:str=''):
		'''
		Method sub with regex

		Params:
			self: instance of object
			pattern (regex pattern, raw string): pattern to look for
			sub_pattern (str): text to replace pattern with
			text (str): text to search
		'''
		pattern = re.compile(pattern)
		matches = pattern.sub(sub_pattern, text)
		# print(matches)
		# matches is a str
		return matches

	def regex_index(self, pattern, text:str):
		'''
		Method search with regex

		Params:
			self: instance of object
			pattern (regex pattern, raw string): pattern to look for
			text (str): text to search
		'''
		pattern = re.compile(pattern)
		matches = pattern.finditer(text)
		for match in matches:
			print(match)
			print(match.group(0))
		return 




        
if __name__ == "__main__":
    print("Executing CorpusProcessor.py")
else:
    print("Importing CorpusProcessor")