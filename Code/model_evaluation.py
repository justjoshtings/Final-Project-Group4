"""
model_evaluation.py
Script to perform modeling evaluation

author: @justjoshtings
created: 4/15/2022
"""
from Woby_Modules.LanguageModel import LanguageModel_GPT2, LanguageModel_GPT_NEO, LanguageModel_GPT2Spooky, CustomTextDatasetGPT2
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from statsmodels.stats.proportion import proportions_ztest

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
gpt2_25_training_stats = pd.read_csv(gpt2_25+'training_stats.csv')

x = np.linspace(1,gpt2_25_training_stats.shape[0],gpt2_25_training_stats.shape[0])

# Loss
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt2_25_training_stats['Training Loss'], color='cadetblue', label='Training')
plt.plot(x, gpt2_25_training_stats['Valid. Loss'], color='tomato', label='Validation')
plt.title('GPT2 Training and Validation Loss', fontsize=20)
plt.legend(loc="upper right")
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Cross Entropy Loss', fontsize=8)
plt.savefig(results_path+'gpt2_25epochs_loss.png', bbox_inches='tight')
plt.clf()
plt.close()

# Perplexity
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt2_25_training_stats['Training Perplexity'], color='cadetblue', label='Training')
plt.plot(x, gpt2_25_training_stats['Valid. Perplexity'], color='tomato', label='Validation')
plt.yscale('log')
plt.ylim(0, 10e2)
plt.title('GPT2 Training and Validation Perplexity', fontsize=20)
plt.legend(loc="upper right")
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Perplexity', fontsize=8)
plt.savefig(results_path+'gpt2_25epochs_perplexity.png', bbox_inches='tight')
plt.clf()
plt.close()

# GPT-NEO 25 Epochs
gpt_neo_25 = './results/model_weights/gpt_neo_125M_25epochs/'
gpt_neo_25_training_stats = pd.read_csv(gpt_neo_25+'training_stats.csv')

x = np.linspace(1,gpt_neo_25_training_stats.shape[0],gpt_neo_25_training_stats.shape[0])

# Loss
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt_neo_25_training_stats['Training Loss'], color='cadetblue', label='Training')
plt.plot(x, gpt_neo_25_training_stats['Valid. Loss'], color='tomato', label='Validation')
plt.title('GPT2 Training and Validation Loss', fontsize=20)
plt.legend(loc="upper right")
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Cross Entropy Loss', fontsize=8)
plt.savefig(results_path+'gpt_neo_25epochs_loss.png', bbox_inches='tight')
plt.clf()
plt.close()

# Perplexity
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt_neo_25_training_stats['Training Perplexity'], color='cadetblue', label='Training')
plt.plot(x, gpt_neo_25_training_stats['Valid. Perplexity'], color='tomato', label='Validation')
plt.yscale('log')
plt.ylim(0, 10e2)
plt.title('GPT2 Training and Validation Perplexity', fontsize=20)
plt.legend(loc="upper right")
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Perplexity', fontsize=8)
plt.savefig(results_path+'gpt_neo_25epochs_perplexity.png', bbox_inches='tight')
plt.clf()
plt.close()

# gpt2spooky 25 Epochs
gpt2spooky_25 = './results/model_weights/gpt2spooky_25epochs/'
gpt2spooky_25_training_stats = pd.read_csv(gpt2spooky_25+'training_stats.csv')

x = np.linspace(1,gpt2spooky_25_training_stats.shape[0],gpt2spooky_25_training_stats.shape[0])

# Loss
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt2spooky_25_training_stats['Training Loss'], color='cadetblue', label='Training')
plt.plot(x, gpt2spooky_25_training_stats['Valid. Loss'], color='tomato', label='Validation')
plt.title('GPT2Spooky Training and Validation Loss', fontsize=20)
plt.legend(loc="upper right")
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Cross Entropy Loss', fontsize=8)
plt.savefig(results_path+'gpt2spooky_25epochs_loss.png', bbox_inches='tight')
plt.clf()
plt.close()

# Perplexity
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (18,6)
plt.plot(x, gpt2spooky_25_training_stats['Training Perplexity'], color='cadetblue', label='Training')
plt.plot(x, gpt2spooky_25_training_stats['Valid. Perplexity'], color='tomato', label='Validation')
plt.yscale('log')
plt.title('GPT2Spooky Training and Validation Perplexity', fontsize=20)
plt.legend(loc="upper right")
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Perplexity', fontsize=8)
plt.savefig(results_path+'gpt2spooky_25epochs_perplexity.png', bbox_inches='tight')
plt.clf()
plt.close()

'''
Load Models to Evaluate
'''
# GPT2 25 Epochs
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

model_gpt2 = LanguageModel_GPT2(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=None,
								valid_data_loader=None,
								test_data_loader=None,
								gpt_model_type='gpt2',
								log_file=SCRAPPER_LOG)

model_gpt2.load_weights(gpt2_25)

# GPT-NEO 25 Epochs
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

model_gpt_neo = LanguageModel_GPT_NEO(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=None,
								valid_data_loader=None,
								test_data_loader=None,
								gpt_model_type='EleutherAI/gpt-neo-125M',
								log_file=SCRAPPER_LOG)

model_gpt_neo.load_weights(gpt_neo_25)

# gpt2spooky 25 Epochs
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('./results/model_weights/gpt2spooky_pretrain/', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

model_gpt2spooky = LanguageModel_GPT2Spooky(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=None,
								valid_data_loader=None,
								test_data_loader=None,
								gpt_model_type='./results/model_weights/gpt2spooky_pretrain/',
								log_file=SCRAPPER_LOG)

model_gpt2spooky.load_weights(gpt2spooky_25)

'''
Generate Text Samples
'''
length = random.randrange(50, 150)
test_sentences = pd.read_csv(CORPUS_FILEPATH+'test_sentences.csv')['0'].values.tolist()

prompts = list()
gpt2_25_generated = list()
gpt_neo_generated = list()
gpt2spooky_generated = list()
gpt2_25_rank = list()
gpt_neo_rank = list()
gpt2spooky_rank = list()

for i in range(1):
	sample = random.choice(test_sentences)
	sample_prompt = ' '.join(sample.split()[:length])
	prompts.append(sample_prompt)

	# GPT2 25 Epochs
	gpt2_generated_text = model_gpt2.generate_text(sample_prompt, max_length=512)
	
	# GPT-NEO 25 Epochs
	gpt_neo_generated_text = model_gpt_neo.generate_text(sample_prompt, max_length=512)

	# gpt2spooky Epochs
	gpt2spooky_generated_text = model_gpt2spooky.generate_text(sample_prompt, max_length=512)

	gpt2_25_generated.append(gpt2_generated_text)
	gpt_neo_generated.append(gpt_neo_generated_text)
	gpt2spooky_generated.append(gpt2spooky_generated_text)

	gpt2_25_rank.append(0)
	gpt_neo_rank.append(0)
	gpt2spooky_rank.append(0)
	
# Prompt | GPT2 25 Generate | GPT-NEO Generate | gpt2spooky Generate | GPT2 25 Rank | GPT-NEO Rank | gpt2spooky Rank |
eval_df = pd.DataFrame(prompts, columns=['prompts'])
eval_df['gpt2_25_generate'] = gpt2_25_generated
eval_df['gpt_neo_25_generate'] = gpt_neo_generated
eval_df['gpt2spooky_generate'] = gpt2spooky_generated
eval_df['gpt2_25_rank'] = gpt2_25_rank
eval_df['gpt_neo_25_rank'] = gpt_neo_rank
eval_df['gpt2spooky_rank'] = gpt2spooky_rank

eval_df.to_csv(results_path+'model_evaluation.csv', index=False)

'''
Plot Evaluation Results
'''
evaluated_results_df = pd.read_excel(results_path+'model_evaluation_checked.xlsx')
evaluated_results_df = evaluated_results_df[~evaluated_results_df['gpt2_25_scary'].isnull()]

print(evaluated_results_df.columns)

# Scary portion
gpt2_scary_portion = evaluated_results_df[evaluated_results_df['gpt2_25_scary'] > 0].sum()['gpt2_25_scary']/evaluated_results_df.shape[0]
gpt_neo_scary_portion = evaluated_results_df[evaluated_results_df['gpt_neo_25_scary'] > 0].sum()['gpt_neo_25_scary']/evaluated_results_df.shape[0]
gpt2spooky_scary_portion = evaluated_results_df[evaluated_results_df['gpt2spooky_scary'] > 0].sum()['gpt2spooky_scary']/evaluated_results_df.shape[0]

# Coherence portion
gpt2_coherent_portion = evaluated_results_df[evaluated_results_df['gpt2_25_choherent'] > 0].sum()['gpt2_25_choherent']/evaluated_results_df.shape[0]
gpt_neo_coherent_portion = evaluated_results_df[evaluated_results_df['gpt_neo_25_coherent'] > 0].sum()['gpt_neo_25_coherent']/evaluated_results_df.shape[0]
gpt2spooky_coherent_portion = evaluated_results_df[evaluated_results_df['gpt2spooky_coherent'] > 0].sum()['gpt2spooky_coherent']/evaluated_results_df.shape[0]

model_cats = ['gpt2', 'gpt-neo', 'gpt2spooky']
scary_portion = [gpt2_scary_portion, gpt_neo_scary_portion, gpt2spooky_scary_portion]
coherent_portion = [gpt2_coherent_portion, gpt_neo_coherent_portion, gpt2spooky_coherent_portion]

# Portion scary
plt.rcParams["figure.figsize"] = (18,6)
plt.bar(model_cats, scary_portion, color='cadetblue')
plt.title('Proportion of Evaluated Stories that are Scary', fontsize=20)
plt.xlabel('Model Evaluated', fontsize=8)
plt.ylabel('Proportion of Evaluated Stories out of 30', fontsize=8)
plt.xticks(rotation='70', fontsize=10)
plt.savefig(results_path+'model_eval_scary_portion.png', bbox_inches='tight')

# Portion coherence
plt.rcParams["figure.figsize"] = (18,6)
plt.bar(model_cats, coherent_portion, color='cadetblue')
plt.title('Proportion of Evaluated Stories that are Coherent', fontsize=20)
plt.xlabel('Model Evaluated', fontsize=8)
plt.ylabel('Proportion of Evaluated Stories out of 30', fontsize=8)
plt.xticks(rotation='70', fontsize=10)
plt.savefig(results_path+'model_eval_coherent_portion.png', bbox_inches='tight')

'''
Hypothesis Testing Model Results
'''
with open(results_path+'hypothesis_testing_results.txt', 'w') as f:
	f.write('')

# 1. gpt2 vs gpt-neo
# 1a. scary
count = np.array(scary_portion[0:2])*evaluated_results_df.shape[0]
nobs = np.array([evaluated_results_df.shape[0],evaluated_results_df.shape[0]])
stat, pval = proportions_ztest(count, nobs)
print('GPT2 vs GPT-NEO Scary Proportion Pval: {0:0.3f}\n'.format(pval))
with open(results_path+'hypothesis_testing_results.txt', 'a') as f:
	f.write('GPT2 vs GPT-NEO Scary Proportion Pval: {0:0.3f}\n'.format(pval))

# 1b. coherence
count = np.array(coherent_portion[0:2])*evaluated_results_df.shape[0]
nobs = np.array([evaluated_results_df.shape[0],evaluated_results_df.shape[0]])
stat, pval = proportions_ztest(count, nobs)
print('GPT2 vs GPT-NEO Coherence Proportion Pval: {0:0.3f}\n'.format(pval))
with open(results_path+'hypothesis_testing_results.txt', 'a') as f:
	f.write('GPT2 vs GPT-NEO Coherence Proportion Pval: {0:0.3f}\n'.format(pval))

# 2. gpt2 vs gpt2spooky
# 2a. scary
count = np.array([scary_portion[0],scary_portion[2]])*evaluated_results_df.shape[0]
nobs = np.array([evaluated_results_df.shape[0],evaluated_results_df.shape[0]])
stat, pval = proportions_ztest(count, nobs)
print('GPT2 vs GPT2Spooky Scary Proportion Pval: {0:0.3f}\n'.format(pval))
with open(results_path+'hypothesis_testing_results.txt', 'a') as f:
	f.write('GPT2 vs GPT2Spooky Scary Proportion Pval: {0:0.3f}\n'.format(pval))
# 2b. coherence
count = np.array([coherent_portion[0],coherent_portion[2]])*evaluated_results_df.shape[0]
nobs = np.array([evaluated_results_df.shape[0],evaluated_results_df.shape[0]])
stat, pval = proportions_ztest(count, nobs)
print('GPT2 vs GPT2Spooky Coherence Proportion Pval: {0:0.3f}\n'.format(pval))
with open(results_path+'hypothesis_testing_results.txt', 'a') as f:
	f.write('GPT2 vs GPT2Spooky Coherence Proportion Pval: {0:0.3f}\n'.format(pval))

# 3. gpt2 vs gpt2spooky
# 3a. scary
count = np.array(scary_portion[1:3])*evaluated_results_df.shape[0]
nobs = np.array([evaluated_results_df.shape[0],evaluated_results_df.shape[0]])
stat, pval = proportions_ztest(count, nobs)
print('GPT-NEO vs GPT2Spooky Scary Proportion Pval: {0:0.3f}\n'.format(pval))
with open(results_path+'hypothesis_testing_results.txt', 'a') as f:
	f.write('GPT-NEO vs GPT2Spooky Scary Proportion Pval: {0:0.3f}\n'.format(pval))
# 3b. coherence
count = np.array(coherent_portion[1:3])*evaluated_results_df.shape[0]
nobs = np.array([evaluated_results_df.shape[0],evaluated_results_df.shape[0]])
stat, pval = proportions_ztest(count, nobs)
print('GPT-NEO vs GPT2Spooky Coherence Proportion Pval: {0:0.3f}\n'.format(pval))
with open(results_path+'hypothesis_testing_results.txt', 'a') as f:
	f.write('GPT-NEO vs GPT2Spooky Coherence Proportion Pval: {0:0.3f}\n'.format(pval))
