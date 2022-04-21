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
import matplotlib.pyplot as plt
import numpy as np
import random

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus_data/'
results_path = './results/'

random_state = 42

# with open('../Woby_Log/my_text.txt') as f:
# 	text = f.read()

'''
Plot trainingcharts
'''
# GPT2 25 Epochs
gpt2_25 = './results/model_weights/gpt2_25epochs/'
gpt2_25_training_stats = pd.read_csv(gpt2_25+'training_stats.csv', index=False)

x = np.linspace(1,gpt2_25_training_stats.shape[0])

# Loss
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt2_25_training_stats['Training Loss'], color='cadetblue')
plt.plot(x, gpt2_25_training_stats['Valid. Loss'], color='tomato')
plt.title('GPT2 Training and Validation Loss', fontsize=20)
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Cross Entropy Loss', fontsize=8)
plt.savefig(results_path+'gpt2_25epochs_loss.png', bbox_inches='tight')

# Perplexity
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt2_25_training_stats['Training Perplexity'], color='cadetblue')
plt.plot(x, gpt2_25_training_stats['Valid. Perplexity'], color='tomato')
plt.title('GPT2 Training and Validation Perplexity', fontsize=20)
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Perplexity', fontsize=8)
plt.savefig(results_path+'gpt2_25epochs_perplexity.png', bbox_inches='tight')

# GPT-NEO 25 Epochs

# ROBERTA X Epochs

'''
Load Models to Evaluate
'''
# GPT2 25 Epochs
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

model_gpt2 = LanguageModel_GPT2(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=None,
								valid_data_loader=None,
								test_data_loader=None,
								gpt_model_type='gpt2',
								log_file=SCRAPPER_LOG)

model_gpt2.load_weights('./results/model_weights/gpt2_10epochs_finetuned/')

# GPT-NEO 25 Epochs

# ROBERTA X Epochs

'''
Generate Text Samples
'''
length = random.randrange(50, 150)
test_sentences = pd.read_csv(CORPUS_FILEPATH+'test_sentences.csv')['0'].values.tolist()

prompts = list()
gpt2_25_generated = list()
gpt2_neo_generated = list()
roberta_generated = list()
gpt2_25_rank = list()
gpt2_neo_rank = list()
roberta_rank = list()

for i in range(50):
	sample = random.choice(test_sentences)
	sample_prompt = ' '.join(sample.split()[:length])
	prompts.append(sample_prompt)

	# GPT2 25 Epochs
	generated_text = model_gpt2.generate_text(sample_prompt)
	
	# GPT-NEO 25 Epochs

	# ROBERTA Epochs

	gpt2_25_generated.append(generated_text)
	gpt2_neo_generated.append()
	roberta_generated.append()

	gpt2_25_rank.append(0)
	gpt2_neo_rank.append(0)
	roberta_rank.append(0)
	

# Prompt | GPT2 25 Generate | GPT-NEO Generate | ROBERTA Generate | GPT2 25 Rank | GPT-NEO Rank | ROBERTA Rank |
eval_df = pd.DataFrame(prompts, columns=['prompts'])
eval_df['gpt2_25_generate'] = gpt2_25_generated
eval_df['gpt_neo_25_generate'] = gpt2_neo_generated
eval_df['roberta_generate'] = roberta_generated
eval_df['gpt2_25_rank'] = gpt2_25_rank
eval_df['gpt_neo_25_rank'] = gpt2_neo_rank
eval_df['roberta_rank'] = roberta_rank

eval_df.to_csv(results_path+'model_evaluation.csv', index=False)
