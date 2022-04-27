"""
test.py
Script to perform some ad-hoc testing of models, not used for application core

author: @justjoshtings
created: 4/15/2022
"""
from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig, LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

model = RobertaForCausalLM.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# tokenizer = ByteLevelBPETokenizer("./results/model_weights/roberta_pretrain/vocab.json", "./results/model_weights/roberta_pretrain/merges.txt",)
# tokenizer._tokenizer.post_processor = BertProcessing(("</s>", tokenizer.token_to_id("</s>")),("<s>", tokenizer.token_to_id("<s>")),)
# tokenizer.enable_truncation(max_length=512)

# inputs = tokenizer.encode("Hello, my dog is cute")
# inputs_ids = torch.tensor(inputs.ids)
# inputs_ids = inputs_ids[None,:]

# print(inputs_ids.shape)

# gen_out = model.generate(input_ids=inputs_ids, max_length=100)
# print(gen_out[0])

# print(tokenizer.decode(gen_out[0], skip_special_tokens=True))

# print(tokenizer.decode(gen_out[0].tolist()))

# outputs = model(**inputs)

prompt = "Hello, my dog is cute, and it is"

# prediction_logits = outputs.logits


# logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),])
# stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
# outputs = model.greedy_search(inputs, logits_processor=logits_processor, stopping_criteria=stopping_criteria)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# print(prediction_logits)

# print(type(tokenizer(prompt, return_tensors="pt")))

# generation_output = model.generate(input_ids=tokenizer.encode(prompt, return_tensors="pt"), max_length=50)

# print(tokenizer.decode(generation_output[0], skip_special_tokens=True))


from transformers import pipeline

fill_mask = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# The sun <mask>.
# =>

result = fill_mask(prompt)
print(result)




# """
# bert_pretraining.py
# Script to perform bert pretraining on our custom corpus

# author: @justjoshtings
# created: 4/20/2022
# """

# import os
# from tokenizers.implementations import  ByteLevelBPETokenizer
# from tokenizers.implementations import BertWordPieceTokenizer
# from transformers import BertConfig
# from transformers import BertTokenizer
# from transformers import BertForMaskedLM
# from transformers import LineByLineTextDataset
# from transformers import DataCollatorForLanguageModeling
# from transformers import Trainer, TrainingArguments
# import pandas as pd
# import json

# PATH = os.getcwd()
# SAVE_MODEL = './results/model_weights/bert_pretrain/'

# if not os.path.exists(SAVE_MODEL):
#     os.makedirs(SAVE_MODEL)

# SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
# CORPUS_FILEPATH = '../corpus_data/'

# train_sentences = pd.read_csv(CORPUS_FILEPATH+'train_sentences.csv')['0'].values.tolist()
# valid_sentences = pd.read_csv(CORPUS_FILEPATH+'valid_sentences.csv')['0'].values.tolist()
# test_sentences = pd.read_csv(CORPUS_FILEPATH+'test_sentences.csv')['0'].values.tolist()

# sentences = train_sentences
# sentences += valid_sentences
# sentences += test_sentences

# corpus_txt = ' '.join(sentences)

# with open(CORPUS_FILEPATH+'corpus_txt.txt', 'w') as f:
#     f.write(corpus_txt)

# # Initialize a tokenizer
# tokenizer = BertWordPieceTokenizer()

# tokenizer.train(files= CORPUS_FILEPATH+"corpus_txt.txt", vocab_size=28000, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# tokenizer.save_model(SAVE_MODEL)

# # tokenizer = BertWordPieceTokenizer(SAVE_MODEL + "/vocab.json", SAVE_MODEL+ "/merges.txt",)

# # tokenizer._tokenizer.post_processor = BertProcessing(("</s>", tokenizer.token_to_id("</s>")),("<s>", tokenizer.token_to_id("<s>")),)

# with open(os.path.join(SAVE_MODEL, "config.json"), "w") as f:
#   tokenizer_cfg = {
#       "do_lower_case": True,
#       "unk_token": "[UNK]",
#       "sep_token": "[SEP]",
#       "pad_token": "[PAD]",
#       "cls_token": "[CLS]",
#       "mask_token": "[MASK]",
#       "model_max_length": 512,
#       "max_len": 512,
#   }
#   json.dump(tokenizer_cfg, f)

# tokenizer.enable_truncation(max_length=512)

# print(tokenizer.encode("For it is in reality vain to profess"))

# config = BertConfig(
#     vocab_size=28000,
#     max_position_embeddings=514,
#     num_attention_heads=12,
#     num_hidden_layers=6,
#     type_vocab_size=1,
# )

# tokenizer = BertTokenizer.from_pretrained(SAVE_MODEL, max_len=512)
# model = BertForMaskedLM(config=config)

# print(model.num_parameters())

# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path= CORPUS_FILEPATH + "/corpus_txt.txt",
#     block_size=128,
# )

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# training_args = TrainingArguments(
#     output_dir= SAVE_MODEL,
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=64,
#     save_steps=10000,
#     save_total_limit=2,
#     prediction_loss_only=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
# )

# trainer.train()
# trainer.save_model(SAVE_MODEL)


