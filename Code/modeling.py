"""
modeling.py
Script to perform modeling

author: @justjoshtings
created: 3/31/2022
"""
from Woby_Modules.LanguageModel import LanguageModel_GPT2, CustomTextDatasetGPT2
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus_data/'

random_state = 42


text = "After a few minutes, I got the notification. I stared at the $700 for at least twenty minutes, expecting to wake up from a dream at any second. But it wasn’t a dream."

# Load Model and Tokenizer
# small, medium, large, xl
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

# model = GPT2LMHeadModel.from_pretrained('gpt2')
# encoded_input = tokenizer.encode(text, return_tensors='pt')

# Generate Output
# sample_outputs = model.generate(input_ids=encoded_input, max_length=150, top_k=50, top_p=0.95, do_sample=True, temperature=0.7, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
# print("Output:\n" + 100 * '-')
# for i, sample_output in enumerate(sample_outputs):
#   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

# Load metadata
corpus_metadadata = pd.read_csv(CORPUS_FILEPATH+'corpus_metadata.csv')
train_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'train']
valid_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'valid']
test_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'test']

# Train Data
# train_corpus_dirs = train_metadata['filepath'][:300]
# train_corpus_dirs = [path.replace('corpus','corpus_data') for path in train_corpus_dirs]
train_sentences = pd.read_csv(CORPUS_FILEPATH+'train_sentences.csv')['0'].values.tolist()
train_data = CustomTextDatasetGPT2(train_sentences, tokenizer)
train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True)

# Validation Data
# valid_corpus_dirs = valid_metadata['filepath'][:300]
# valid_corpus_dirs = [path.replace('corpus','corpus_data') for path in valid_corpus_dirs]
valid_sentences = pd.read_csv(CORPUS_FILEPATH+'valid_sentences.csv')['0'].values.tolist()
valid_data = CustomTextDatasetGPT2(valid_sentences, tokenizer)
valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=True)

# Test Data
# test_corpus_dirs = test_metadata['filepath'][:300]
# test_corpus_dirs = [path.replace('corpus','corpus_data') for path in test_corpus_dirs]
test_sentences = pd.read_csv(CORPUS_FILEPATH+'test_sentences.csv')['0'].values.tolist()
test_data = CustomTextDatasetGPT2(test_sentences, tokenizer)
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True)

print('Num Train: ', len(train_data_loader), 
	'Num Validation: ', len(valid_data_loader), 
	'Num Test: ', len(test_data_loader), 
	'Total Num: ', len(train_data_loader)+len(valid_data_loader)+len(test_data_loader))

model_gpt2 = LanguageModel_GPT2(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=train_data_loader,
								valid_data_loader=valid_data_loader,
								test_data_loader=test_data_loader,
								gpt_model_type='gpt2',
								log_file=SCRAPPER_LOG)

# model_gpt2.generate_text(text)
model_gpt2.train(num_epochs=1, model_weights_dir='./results/model_weights/gpt2_01/')
model_gpt2.get_training_stats(model_weights_dir='./results/model_weights/gpt2_01/training_stats.csv')


# accuracy_metric = load_metric('accuracy')
# bertscore_metric = load_metric('bertscore')
# bleu_metric = load_metric('bleu')
# f1_metric = load_metric('f1')
# meteor_metric = load_metric('meteor')
# precision_metric = load_metric('precision')
# recall_metric = load_metric('recall')
# rouge_metric = load_metric('rouge')