"""
LanguageModel.py
Object to handle language models for Woby.

author: @justjoshtings
created: 4/8/2022
"""
from operator import index
from Woby_Modules.Logger import MyLogger
from transformers import GPT2Tokenizer, GPT2Model, pipeline, set_seed, GPT2LMHeadModel
from transformers import get_scheduler
import datasets
from datasets import load_metric
import torch, gc
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
from tqdm.auto import tqdm
import time
from datetime import datetime
import datetime as dt
import random
import os
import math

class LanguageModel:
	'''
	Object to handle language models for Woby.
	'''
	def __init__(self, corpus_filepath, random_state, train_data_loader, valid_data_loader, test_data_loader, log_file=None):
		'''
		Params:
			self: instance of object
			corpus_filepath (str): corpus filepath to save text data to, './corpus/'
			random_state (int): random seed
			train_data_loader (torch.utils.data.DataLoader): train data loader
			valid_data_loader (torch.utils.data.DataLoader): validation data loader
			test_data_loader (torch.utils.data.DataLoader): test data loader
			log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
		'''
		self.corpus_filepath = corpus_filepath
		self.random_state = random_state
		self.train_data_loader = train_data_loader
		self.valid_data_loader = valid_data_loader
		self.test_data_loader = test_data_loader

		self.LOG_FILENAME = log_file
		if self.LOG_FILENAME:
            # Set up a specific logger with our desired output level
			self.mylogger = MyLogger(self.LOG_FILENAME)
			# global MY_LOGGER
			self.MY_LOGGER = self.mylogger.get_mylogger()
			self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel] Setting corpus filepath to {self.corpus_filepath}...")
	
	def save_weights(self):
		'''
		Method to save model weights

		Params:
			self: instance of object
		'''
		# Create output directory if needed
		if not os.path.exists(self.model_weights_dir):
			os.makedirs(self.model_weights_dir)

		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel] Saving model to {self.model_weights_dir}...")
		print("Saving model to %s" % self.model_weights_dir)
		
		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(self.model_weights_dir)
		self.tokenizer.save_pretrained(self.model_weights_dir)

		# Good practice: save your training arguments together with the trained model
		# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
	
	def format_time(self, elapsed):
		'''
		Method to format time

		Params:
			self: instance of object
			elapsed (float): time elapsed
		
		Returns:
			elapsed time in str
		'''
		return str(dt.timedelta(seconds=int(round((elapsed)))))

class LanguageModel_GPT2(LanguageModel):
	'''
	GPT2 Model
	'''
	def __init__(self, corpus_filepath, random_state, train_data_loader, valid_data_loader, test_data_loader, gpt_model_type='gpt2', log_file=None):
		'''
		Params:
			self: instance of object
			corpus_filepath (str): corpus filepath to save text data to, './corpus/'
			random_state (int): random seed
			train_data_loader (torch.utils.data.DataLoader): train data loader
			valid_data_loader (torch.utils.data.DataLoader): validation data loader
			test_data_loader (torch.utils.data.DataLoader): test data loader
			gpt_model_type (str): ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
			log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log
		'''
		LanguageModel.__init__(self, corpus_filepath, random_state, train_data_loader, valid_data_loader, test_data_loader, log_file=log_file)
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
		self.model = GPT2LMHeadModel.from_pretrained('gpt2')

	def generate_text(self, prompt, max_length=150, top_k=50, top_p=0.95, do_sample=True, temperature=0.7, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True):
		'''
		Function to generate text from given prompt

		Params:
			self: instance of object
			prompt (str): string input prompt
			max_length (int): max length of generated output, default = 150
			top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering, default = 50
			top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation, default = 0.95
			do_sample (Boolean): Whether or not to use sampling ; use greedy decoding otherwise., default = True
			temperature (float): The value used to module the next token probabilities, default = 0.7
			num_return_sequences (int): The number of independently computed returned sequences for each element in the batch., default = 1
			no_repeat_ngram_size (int): If set to int > 0, all ngrams of that size can only occur once., default = 2
			early_stopping (Boolean):  Whether to stop the beam search when at least num_beams sentences are finished per batch or not, default = True
		'''
		generated_encoded_texts = self.model.generate(input_ids=self.tokenizer.encode(prompt, return_tensors='pt'), 
											max_length=max_length, 
											top_k=50, 
											top_p=0.95, 
											do_sample=True, 
											temperature=0.7, 
											num_return_sequences=1, 
											no_repeat_ngram_size=2, 
											early_stopping=True
											) 
		print("Output:\n" + 100 * '-')
		for i, encoded_text in enumerate(generated_encoded_texts):
			print("{}: {}".format(i, self.tokenizer.decode(encoded_text, skip_special_tokens=True)))
	
	def train(self, num_epochs=1, lr=5e-5, sample_every=100, eval_during_training=False, save_weights=True, model_weights_dir='./results/model_weights/') :
		'''
		Method to perform training loop

		Params:
			self: instance of object
			num_epochs (int): number of epochs to train, default=1
			lr (float): learning rate, default=5e-5
			sample_every (int): number of training steps to print progress, default=100
			eval_during_training (Boolean): whether to evaluate during training every sample_every, default=False
			save_weights (Boolean): whether to save weights or not after training
			model_weights_dir (str): directory to save model weights after model training completion
		
		# Training Loop: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=gpt6tR83keZD
		'''
		# Optimizer and Learning Rate Scheduler
		optimizer = AdamW(self.model.parameters(), lr=lr)

		num_epochs = num_epochs
		num_training_steps = num_epochs * len(self.train_data_loader)
		lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

		self.model.resize_token_embeddings(len(self.tokenizer))
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		
		print('Using device..', device)
		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] Using device {device}...")	

		self.model.to(device)

		progress_bar = tqdm(range(num_training_steps))

		total_t0 = time.time()
		self.training_stats = []
		sample_every = sample_every

		gc.collect()
		torch.cuda.empty_cache()

		for epoch in range(num_epochs):
			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
			print('Training...')
			if self.LOG_FILENAME:
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] \n======== Epoch {epoch + 1} / {num_epochs} ========")	
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] Training...")	
			
			total_train_loss = 0
			total_train_perplexity = 0

			t0 = time.time()
			self.model.train()

			for step, batch in enumerate(self.train_data_loader):
				b_input_ids = batch[0].to(device)
				b_labels = batch[0].to(device)
				b_masks = batch[1].to(device)

				self.model.zero_grad()        

				outputs = self.model(b_input_ids,
									labels=b_labels, 
									attention_mask = b_masks,
									token_type_ids=None
								)

				loss = outputs[0]

				batch_loss = loss.item()
				batch_perplexity = math.exp(batch_loss)
				
				total_train_loss += batch_loss
				total_train_perplexity += batch_perplexity
				 
				# Get sample every x batches.
				if step % sample_every == 0 and not step == 0:

					elapsed = self.format_time(time.time() - t0)
					print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(self.train_data_loader), batch_loss, elapsed))
					if self.LOG_FILENAME:
						self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]   Batch {step}  of  {len(self.train_data_loader)}. Loss: {batch_loss}.   Elapsed: {elapsed}.")	

					if eval_during_training:
						self.model.eval()

						sample_outputs = self.model.generate(
												bos_token_id=random.randint(1,30000),
												do_sample=True,   
												top_k=50, 
												max_length = 200,
												top_p=0.95, 
												num_return_sequences=1,
												no_repeat_ngram_size=2,
												early_stopping=True
											)
						for i, sample_output in enumerate(sample_outputs):
							print("{}: {}".format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True)))
							if self.LOG_FILENAME:
								self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] {i}: {self.tokenizer.decode(sample_output, skip_special_tokens=True)}")	
						
						self.model.train()

				loss.backward()
				optimizer.step()
				lr_scheduler.step()
				progress_bar.update(1)

			# Calculate the average loss over all of the batches.
			avg_train_loss = total_train_loss / len(self.train_data_loader)       
			avg_train_perplexity = total_train_perplexity / len(self.train_data_loader)       

			# Measure how long this epoch took.
			training_time = self.format_time(time.time() - t0)

			print("")
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
			print("  Average training perplexity: {0:.2f}".format(avg_train_perplexity))
			print("  Training epoch took: {:}".format(training_time))
			if self.LOG_FILENAME:
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]\n  Average training loss: {avg_train_loss}")	
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]\n  Average training perplexity: {avg_train_perplexity}")	
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training]  Training epoch took: {training_time}")	
				
			# ========================================
			#               Validation
			# ========================================

			print("")
			print("Running Validation...")
			if self.LOG_FILENAME:
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Validation] Running Validation...")	

			t0 = time.time()

			self.model.eval()

			total_eval_loss = 0
			total_eval_perplexity = 0
			nb_eval_steps = 0

			# Evaluate data for one epoch
			for batch in self.valid_data_loader:
				
				b_input_ids = batch[0].to(device)
				b_labels = batch[0].to(device)
				b_masks = batch[1].to(device)
				b_next_input_ids = batch[2].to(device)

				with torch.no_grad():        

					outputs  = self.model(b_input_ids, 
		#                            token_type_ids=None, 
									attention_mask = b_masks,
									labels=b_labels)
				
					loss = outputs[0]
					
				batch_loss = loss.item()
				batch_perplexity = math.exp(batch_loss)

				total_eval_loss += batch_loss
				total_eval_perplexity += batch_perplexity


			avg_val_loss = total_eval_loss / len(self.valid_data_loader)
			avg_val_perplexity = total_eval_perplexity / len(self.valid_data_loader)
			
			validation_time = self.format_time(time.time() - t0)    

			print("  Validation Loss: {0:.2f}".format(avg_val_loss))
			print("  Validation Perplexity: {0:.2f}".format(avg_val_perplexity))
			print("  Validation took: {:}".format(validation_time))
			if self.LOG_FILENAME:
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Validation]   Validation Loss: {avg_val_loss}")	
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Validation]   Validation Perplexity: {avg_val_perplexity}")	
				self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Validation]     Validation took: {validation_time}")	

			# Record all statistics from this epoch.
			self.training_stats.append(
				{
					'epoch': epoch + 1,
					'Training Loss': avg_train_loss,
					'Training Perplexity': avg_train_perplexity,
					'Valid. Loss': avg_val_loss,
					'Valid. Perplexity': avg_val_perplexity,
					'Training Time': training_time,
					'Validation Time': validation_time
				}
			)

		print("")
		print("Training complete!")
		print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
		if self.LOG_FILENAME:
			self.MY_LOGGER.info(f"{datetime.now()} -- [LanguageModel Training] Training complete!\n Total training took {self.format_time(time.time()-total_t0)} (h:mm:ss)")
		
		if save_weights:
			self.model_weights_dir = model_weights_dir
			self.save_weights()

	def get_training_stats(self, save_weights=True, model_weights_dir='./results/model_weights/training_stats.csv'):
		'''
		Method to get trainig stats
		
		Params:
			self: instance of object	
		Returns:
			df_stats (pandas df): trainig stats	
		'''
		# Create a DataFrame from our training statistics.
		df_stats = pd.DataFrame(data=self.training_stats)

		# Use the 'epoch' as the row index.
		df_stats = df_stats.set_index('epoch')

		if save_weights:
			df_stats.to_csv(model_weights_dir, index=False)

		# Display the table.
		print(df_stats.head(100))

		return df_stats

	def load_weights(self):
		'''
		Method to save model weights
		
		Params:
			self: instance of object
		
		Returns:
			model (torch model): loaded model

		'''
		# Load a trained model and vocabulary that you have fine-tuned
		self.model = GPT2LMHeadModel.from_pretrained(self.model_weights_dir)
		self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_weights_dir)

		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		print('Using device..', device)
		self.model.to(device)

		return self.model


class CustomTextDataset(Dataset):
	'''
	CustomTextDataset object
	'''
	def __init__(self, sentences_list, tokenizer):
		'''
		Params:
			self: instance of object
			sentences_list (list of str): list of sentences
			tokenizer (tokenizer object): tokenizer function
		'''
		self.sentences_list = sentences_list
		self.tokenizer = tokenizer
	
	def __len__(self):
		'''
		Params:
			self: instance of object
		Returns:
			number of corpus texts
		'''
		return len(self.sentences_list)

class CustomTextDatasetGPT2(CustomTextDataset):
	'''
	CustomTextDataset object for GPT2
	'''
	def __init__(self, sentences_list, tokenizer, gpt2_type="gpt2", return_tensors_type="pt", max_length=768):
		'''
		Params:
			self: instance of object
			sentences_list (list of str): list of sentences
			tokenizer (tokenizer object): tokenizer function
			gpt2_type (str): ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2'
			return_tensors_type (str): ['pt', 'tf'], default='pt'
			max_length (int): max length for tokenizer input, default=768 
		'''
		CustomTextDataset.__init__(self, sentences_list, tokenizer)

		self.max_length = max_length
		self.gpt2_type = gpt2_type
		self.return_tensors_type = return_tensors_type

	def __getitem__(self, idx):
		'''
		Params:
			self: instance of object
			idx (int): index of iteration
		Returns:
			input_ids (pt tensors): encoded text as tensors
			attn_masks (pt tensors): attention masks as tensors
		'''
		text = self.sentences_list[idx]
		encodings_dict = self.tokenizer('<|startoftext|>'+ text + '<|endoftext|>', truncation=True, max_length=self.max_length, padding="max_length")
		input_ids = torch.tensor(encodings_dict['input_ids'])
		attn_masks = torch.tensor(encodings_dict['attention_mask'])
		
		try:
			next_text = self.sentences_list[idx+1]
		except IndexError:
			next_text = text
		next_encodings_dict = self.tokenizer('<|startoftext|>'+ next_text + '<|endoftext|>', truncation=True, max_length=self.max_length, padding="max_length")
		next_input_ids = torch.tensor(next_encodings_dict['input_ids'])
		
		return input_ids, attn_masks, next_input_ids

if __name__ == "__main__":
    print("Executing LanguageModel.py")
else:
    print("Importing LanguageModel")