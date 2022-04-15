"""
modeling.py
Script to perform modeling

author: @justjoshtings
created: 4/15/2022
"""
from Woby_Modules.LanguageModel import LanguageModel_GPT2, LanguageModel_GPT_NEO, CustomTextDatasetGPT2
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus_data/'

random_state = 42

with open('../Woby_Log/my_text.txt') as f:
	text = f.read()

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

model_gpt2 = LanguageModel_GPT2(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=None,
								valid_data_loader=None,
								test_data_loader=None,
								gpt_model_type='gpt2',
								log_file=SCRAPPER_LOG)

model_gpt2.load_weights('./results/model_weights/gpt2_finetuned/')
model_gpt2.generate_text(text)
