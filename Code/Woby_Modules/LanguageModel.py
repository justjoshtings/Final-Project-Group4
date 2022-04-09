"""
LanguageModel.py
Object to handle language models for Woby.

author: @justjoshtings
created: 4/8/2022
"""
from Woby_Modules.Logger import MyLogger
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model

class LanguageModel:
	'''
	Object to handle language models for Woby.
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
			self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel] Setting corpus filepath to {self.corpus_filepath}...")


class LanguageModel_GPT2(LanguageModel):
	'''
	GPT2 Model
	'''
	def __init__(self, corpus_filepath, log_file=None):
		'''
		Params:
			self: instance of object
			log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
			corpus_filepath (str): corpus filepath to save text data to, './corpus/'
		'''
		LanguageModel.__init__(self, corpus_filepath, log_file=log_file)
