"""
modeling.py
Script to perform modeling

author: @justjoshtings
created: 3/31/2022
"""
import os
from Woby_Modules.LanguageModel import LanguageModel, LanguageModel_GPT2, CustomTextDataset, CustomTextDatasetGPT2
from transformers import GPT2Tokenizer, GPT2Model, pipeline, set_seed, GPT2LMHeadModel
from transformers import get_scheduler
import datasets
import torch, gc
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
from tqdm.auto import tqdm
import time
import datetime
import random

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus_data/'

# model = LanguageModel_GPT2(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

set_seed(42)

text = "After a few minutes, I got the notification. I stared at the $700 for at least twenty minutes, expecting to wake up from a dream at any second. But it wasn’t a dream."

# Load Model and Tokenizer
# small, medium, large, xl
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2')
encoded_input = tokenizer.encode(text, return_tensors='pt')

# Generate Output
# sample_outputs = model.generate(input_ids=encoded_input, max_length=150, top_k=50, top_p=0.95, do_sample=True, temperature=0.7, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
# print("Output:\n" + 100 * '-')
# for i, sample_output in enumerate(sample_outputs):
#   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

# Load metadata
corpus_metadadata = pd.read_csv(CORPUS_FILEPATH+'corpus_metadata.csv')
train_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'train']
test_metadata = corpus_metadadata[corpus_metadadata['train_test'] == 'test']

# Train Data
train_corpus_dirs = train_metadata['filepath'][:300]
train_corpus_dirs = [path.replace('corpus','corpus_data') for path in train_corpus_dirs]
train_data = CustomTextDatasetGPT2(train_corpus_dirs, tokenizer)
train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True)

# for (idx, batch) in enumerate(train_data_loader):
  # # Print the 'text' data of the batch
  # print(idx, batch[0], '\n\n\n')

# Test Data
test_corpus_dirs = test_metadata['filepath'][:300]
test_corpus_dirs = [path.replace('corpus','corpus_data') for path in test_corpus_dirs]
test_data = CustomTextDatasetGPT2(test_corpus_dirs, tokenizer)
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# Optimizer and Learning Rate Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_data_loader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Using device..', device)
model.to(device)

# Training Loop: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=gpt6tR83keZD
progress_bar = tqdm(range(num_training_steps))

total_t0 = time.time()
training_stats = []
sample_every = 100

gc.collect()
torch.cuda.empty_cache()

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

for epoch in range(num_epochs):
	print("")
	print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
	print('Training...')
	total_train_loss = 0
	t0 = time.time()
	model.train()

	for step, batch in enumerate(train_data_loader):
		b_input_ids = batch[0].to(device)
		b_labels = batch[0].to(device)
		b_masks = batch[1].to(device)

		model.zero_grad()        

		outputs = model(b_input_ids,
							labels=b_labels, 
							attention_mask = b_masks,
							token_type_ids=None
						)

		loss = outputs[0]

		batch_loss = loss.item()
		total_train_loss += batch_loss
        
		# Get sample every x batches.
		if step % sample_every == 0 and not step == 0:

			elapsed = format_time(time.time() - t0)
			print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_data_loader), batch_loss, elapsed))

		# 	model.eval()

		# 	sample_outputs = model.generate(
		# 							bos_token_id=random.randint(1,30000),
		# 							do_sample=True,   
		# 							top_k=50, 
		# 							max_length = 200,
		# 							top_p=0.95, 
		# 							num_return_sequences=1,
		# 							no_repeat_ngram_size=2,
		# 							early_stopping=True
		# 						)
		# 	for i, sample_output in enumerate(sample_outputs):
		# 		print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
		# 	model.train()

		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		progress_bar.update(1)

	# Calculate the average loss over all of the batches.
	avg_train_loss = total_train_loss / len(train_data_loader)       

	# Measure how long this epoch took.
	training_time = format_time(time.time() - t0)

	print("")
	print("  Average training loss: {0:.2f}".format(avg_train_loss))
	print("  Training epoch took: {:}".format(training_time))
		
	# ========================================
	#               Validation
	# ========================================

	print("")
	print("Running Validation...")

	t0 = time.time()

	model.eval()

	total_eval_loss = 0
	nb_eval_steps = 0

    # Evaluate data for one epoch
	for batch in test_data_loader:
        
		b_input_ids = batch[0].to(device)
		b_labels = batch[0].to(device)
		b_masks = batch[1].to(device)

		with torch.no_grad():        

			outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
			loss = outputs[0]  
            
		batch_loss = loss.item()
		total_eval_loss += batch_loss        

	avg_val_loss = total_eval_loss / len(test_data_loader)
    
	validation_time = format_time(time.time() - t0)    

	print("  Validation Loss: {0:.2f}".format(avg_val_loss))
	print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
	training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
      

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# Display the table.
df_stats.head()