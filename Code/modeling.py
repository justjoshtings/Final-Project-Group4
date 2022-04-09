"""
modeling.py
Script to perform modeling

author: @justjoshtings
created: 3/31/2022
"""
import os
from Woby_Modules.LanguageModel import LanguageModel, LanguageModel_GPT2
from transformers import GPT2Tokenizer, GPT2Model, pipeline, set_seed, GPT2LMHeadModel

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus_data/'

# model = LanguageModel_GPT2(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

set_seed(42)

text = "After a few minutes, I got the notification. I stared at the $700 for at least twenty minutes, expecting to wake up from a dream at any second. But it wasn’t a dream."

# small, medium, large, xl
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
encoded_input = tokenizer.encode(text, return_tensors='pt')

sample_outputs = model.generate(input_ids=encoded_input, max_length=150, top_k=50, top_p=0.95, do_sample=True, temperature=0.7, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


