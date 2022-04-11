"""
LanguageModel.py
Object to handle language models for Woby.

author: @justjoshtings
created: 4/8/2022
"""
from Woby_Modules.Logger import MyLogger
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
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


class CustomTextDataset(Dataset):
	def __init__(self, corpus_dirs, tokenizer):
		self.corpus_dirs = corpus_dirs
		self.tokenizer = tokenizer
	
	def __len__(self):
		return len(self.corpus_dirs)

class CustomTextDatasetGPT2(CustomTextDataset):
	def __init__(self, corpus_dirs, tokenizer, gpt2_type="gpt2", return_tensors_type="pt", max_length=768):
		CustomTextDataset.__init__(self, corpus_dirs, tokenizer)

		self.max_length = max_length
		self.gpt2_type = gpt2_type
		self.return_tensors_type = return_tensors_type

	def __getitem__(self, idx):
		with open(self.corpus_dirs[idx], 'r+') as f:
			text = f.read()

		encodings_dict = self.tokenizer('<|startoftext|>'+ text + '<|endoftext|>', truncation=True, max_length=self.max_length, padding="max_length")
		input_ids = torch.tensor(encodings_dict['input_ids'])
		attn_masks = torch.tensor(encodings_dict['attention_mask'])
		
		return input_ids, attn_masks

if __name__ == "__main__":
    print("Executing LanguageModel.py")
else:
    print("Importing LanguageModel")