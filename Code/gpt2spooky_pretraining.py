"""
gpt2spooky_pretraining.py
Script to perform gpt2spooky pretraining on our custom corpus

author: @justjoshtings
created: 4/20/2022
"""

import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import GPT2Config
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from pathlib import Path
import pandas as pd

PATH = os.getcwd()
SAVE_MODEL = './results/model_weights/gpt2spooky_pretrain/'

if not os.path.exists(SAVE_MODEL):
    os.makedirs(SAVE_MODEL)

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus_data/'

train_sentences = pd.read_csv(CORPUS_FILEPATH+'train_sentences.csv')['0'].values.tolist()
valid_sentences = pd.read_csv(CORPUS_FILEPATH+'valid_sentences.csv')['0'].values.tolist()
test_sentences = pd.read_csv(CORPUS_FILEPATH+'test_sentences.csv')['0'].values.tolist()

sentences = train_sentences
sentences += valid_sentences
sentences += test_sentences

corpus_txt = ' '.join(sentences)

with open(CORPUS_FILEPATH+'corpus_txt.txt', 'w') as f:
    f.write(corpus_txt)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

#tokenizer.train(files=paths, vocab_size=28000, min_frequency=2, special_tokens=["<s>","<pad>", "</s>","<unk>", "<mask>",])
tokenizer.train(files= CORPUS_FILEPATH+"corpus_txt.txt", vocab_size=8000, min_frequency=2, special_tokens=["<|startoftext|>","<|endoftext|>", "<|pad|>",])

tokenizer.save_model(SAVE_MODEL)

tokenizer = ByteLevelBPETokenizer(SAVE_MODEL + "/vocab.json", SAVE_MODEL+ "/merges.txt",)

tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("For it is in reality vain to profess"))

config = GPT2Config(
    vocab_size=8000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = GPT2TokenizerFast.from_pretrained(SAVE_MODEL, max_len=512, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>', )

model = GPT2LMHeadModel(config=config)

print(model.num_parameters())

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path= CORPUS_FILEPATH + "/corpus_txt.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir= SAVE_MODEL,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(SAVE_MODEL)


