"""
modeling.py
Script to perform modeling

author: @justjoshtings
created: 3/31/2022
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

'''
Load tokenizer, metadata, and data loaders
'''
# Load Model and Tokenizer
# small, medium, large, xl
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

# Load metadata
corpus_metadadata = pd.read_csv(CORPUS_FILEPATH+'corpus_metadata.csv')
train_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'train']
valid_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'valid']
test_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'test']

# Train Data
# train_corpus_dirs = train_metadata['filepath']
# train_corpus_dirs = [path.replace('corpus','corpus_data') for path in train_corpus_dirs]
train_sentences = pd.read_csv(CORPUS_FILEPATH+'train_sentences.csv')['0'].values.tolist()

# Validation Data
# valid_corpus_dirs = valid_metadata['filepath']
# valid_corpus_dirs = [path.replace('corpus','corpus_data') for path in valid_corpus_dirs]
valid_sentences = pd.read_csv(CORPUS_FILEPATH+'valid_sentences.csv')['0'].values.tolist()

# Test Data
# test_corpus_dirs = test_metadata['filepath']
# test_corpus_dirs = [path.replace('corpus','corpus_data') for path in test_corpus_dirs]
test_sentences = pd.read_csv(CORPUS_FILEPATH+'test_sentences.csv')['0'].values.tolist()


'''
GPT2 Model Training
'''
# Train
gpt2_train_data = CustomTextDatasetGPT2(train_sentences, gpt2_tokenizer, max_length=768)
gpt2_train_data_loader = DataLoader(gpt2_train_data, batch_size=1, shuffle=True)
# Valid
gpt2_valid_data = CustomTextDatasetGPT2(valid_sentences, gpt2_tokenizer, max_length=768)
gpt2_valid_data_loader = DataLoader(gpt2_valid_data, batch_size=1, shuffle=True)
# Test
gpt2_test_data = CustomTextDatasetGPT2(test_sentences, gpt2_tokenizer, max_length=768)
gpt2_test_data_loader = DataLoader(gpt2_test_data, batch_size=1, shuffle=True)

print('Num Train: ', len(gpt2_train_data_loader), 
	'Num Validation: ', len(gpt2_valid_data_loader), 
	'Num Test: ', len(gpt2_test_data_loader), 
	'Total Num: ', len(gpt2_train_data_loader)+len(gpt2_valid_data_loader)+len(gpt2_test_data_loader))

model_gpt2 = LanguageModel_GPT2(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=gpt2_train_data_loader,
								valid_data_loader=gpt2_valid_data_loader,
								test_data_loader=gpt2_test_data_loader,
								gpt_model_type='gpt2',
								log_file=SCRAPPER_LOG)

# model_gpt2.generate_text(text)
model_gpt2.train(num_epochs=25, model_weights_dir='./results/model_weights/gpt2_25epochs/')
model_gpt2.get_training_stats(model_weights_dir='./results/model_weights/gpt2_25epochs/training_stats.csv')

'''
GPT NEO Model Training
'''
# Train
gpt_neo_train_data = CustomTextDatasetGPT2(train_sentences, gpt_neo_tokenizer, max_length=1024)
gpt_neo_train_data_loader = DataLoader(gpt_neo_train_data, batch_size=1, shuffle=True)
# Valid
gpt_neo_valid_data = CustomTextDatasetGPT2(valid_sentences, gpt2_tokenizer, max_length=1024)
gpt_neo_valid_data_loader = DataLoader(gpt_neo_valid_data, batch_size=1, shuffle=True)
# Test
gpt_neo_test_data = CustomTextDatasetGPT2(test_sentences, gpt2_tokenizer, max_length=1024)
gpt_neo_test_data_loader = DataLoader(gpt_neo_test_data, batch_size=1, shuffle=True)

print('Num Train: ', len(gpt_neo_train_data_loader), 
	'Num Validation: ', len(gpt_neo_valid_data_loader), 
	'Num Test: ', len(gpt_neo_test_data_loader), 
	'Total Num: ', len(gpt_neo_train_data_loader)+len(gpt_neo_valid_data_loader)+len(gpt_neo_test_data_loader))

model_gpt_neo = LanguageModel_GPT_NEO(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=gpt_neo_train_data_loader,
								valid_data_loader=gpt_neo_valid_data_loader,
								test_data_loader=gpt_neo_test_data_loader,
								gpt_model_type='EleutherAI/gpt-neo-125M',
								log_file=SCRAPPER_LOG)

# model_gpt_neo.generate_text(text)
model_gpt_neo.train(num_epochs=25, model_weights_dir='./results/model_weights/gpt_neo_125M_25epochs/')
model_gpt_neo.get_training_stats(model_weights_dir='./results/model_weights/gpt_neo_125M_25epochs/training_stats.csv')


# from transformers import GPT2Tokenizer, GPT2Model, pipeline, set_seed, GPT2LMHeadModel
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

# sizes = list()
# for train in train_corpus_dirs:
# 	with open(train, 'r') as f:
# 		text = f.read()

# 	encodings_dict = tokenizer('<|startoftext|>'+ text + '<|endoftext|>', truncation=True, max_length=768, padding="max_length")
# 	my_size = len([n for n in encodings_dict['input_ids'] if n != 50258])
# 	sizes.append(my_size)

# print('Mean', sum(sizes) / len(sizes))
# print('Max', max(sizes))
# print('Min', min(sizes))
# print(len(sizes))
# print(len([size for size in sizes if size == 768]))