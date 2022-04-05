## To Do
#### Preprocessing
* EDA (LDA, )
      plots:
            * stories per subreddit
            * most common words total + by each subreddit
            * avg number of words per sub
            * LDA
                  * network plot
                  * topics plot

#### Model Pipeline
* build simple nn in pytorch for next word prediction
* figure out pipeline for gpt2, gpt neo, bert, custom model
      * What inputs are needed for each model?
      * tokenizer will need to handle internet speech/emojis

#### Model Training/Finetuning
1. Basic transformer
2. GPT-2 (no finetune)
3. GPT-2 (finetuned)
4. GPT-NEO (no finetune)
5. GPT-NEO (finetuned)
6. BERT (no finetune)
7. BERT (finetuned)

#### Model Evaluation
* scores: bleau, rouge, perplexity, meteor, bertscore


## Execution
Options:
1. Scrape reddit, push to kaggle, pull from kaggle, preprocess, ... [not real option]
2. Pull from kaggle, preprocess, ...
3. Load from trained weights, ...

## Contents
1. sample_results
2. Woby_keys
3. Woby_Modules
4.

# <a name="data-acquisition"></a>
## Data Acquisition

Data Acquisition from posts of different sub-reddits were done with two primary services:

1. [Reddit Dev API](https://www.reddit.com/dev/api/) with **requests**
2. [Python Pushshift.io API Wrapper (PSAW)](https://psaw.readthedocs.io/en/latest/)

The **Reddit Dev API** has a limit of 1000 Reddit post submissions it can GET. **PSAW** theorectically has all historical post submissions but it has issues where some shards may be out of service and therefore not all post submissions can be retrieved. I used a combination of both services then removed duplciates in order to get the largest corpus possible. 

The **Reddit Dev API** and **PSAW** returned a total of 23,249 posts from the subreddits. This final number becomes 14,714 after removing duplicated stories and 137MBs.

For each post retrieved, the JSON response was parsed for metadata which was inserted into a MongoDB database while a copy of the submission text was saved into the **corpus** directory organized by sub-reddit with a [doc_id]_[t3]_[reddit_post_id].txt schema.

#### MongoDB stories' metadata sample schema

```
data_dict = {
            'doc_id': ,
            'full_name': ,
            'subreddit': ,
            'subreddit_name_prefixed': ,
            'title': ,
            'little_taste': ,
            'selftext': ,
            'author': ,
            'upvote_ratio': ,
            'ups': ,
            'downs': ,
            'score': ,
            'num_comments': ,
            'permalink': ,
            'kind': ,
            'num_characters': ,
            'num_bytes': ,
            'created_utc': ,
            'created_human_readable': ,
            'filepath': ,
            'train_test': 
      }
```

#### Corpus directory setup
```
FINAL-PROJECT-GROUP4
│─── Code
|─── Woby_Log
|─── ...
│
└─── Corpus
│    │
│    └───nosleep
│    │    │ 1_t3_diuucz.txt
│    │    │ 2_t3_dyqd5e.txt
│    │    │ ...
│    │
│    └───creepyencounters
│    │    │ 5931_t3_i3l009.txt
│    │    │ 5931_t3_i3l009.txt
│    │    │ ...
│    │
│    └───Ghoststories
│    │    │ 9845_t3_jdedeb.txt
│    │    │ 9846_t3_hvu2ko.txt
│    │    │ ...
```

#### More on Data Acquisition:
* [Python Reddit API Wrapper (PRAW)](https://praw.readthedocs.io/en/stable/) can also be used in place of the **Reddit Dev API** to abstract away the requests code.
* [How to Use Reddit API in Python](https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c)

# <a name="data-download"></a>
## Data Distribution and Download

Data can be accessed publicly on [Kaggle](https://www.kaggle.com/datasets/justjoshtings/spooky-reddit-stories). It conists of a **corpus_metadata.csv** and each individual subreddits' stories in .txt format.

**Two options to download data:**

**Option 1:** Manual Download
1. Manually download, unzip, and move all contents to **FINAL-PROJECT-GROUP4/corpus_data/**.

**Option 2:** Use Kaggle API to download data
1. Create a Kaggle account API. See [here](https://github.com/Kaggle/kaggle-api#api-credentials) or [here](https://adityashrm21.github.io/Setting-Up-Kaggle/).
2. Download the kaggle.json file of your API credentials and save to **~/.kaggle/kaggle.json**
```
mv [downloaded kaggle.json path] ~/.kaggle/kaggle.json
```
3. Set permissions.
```
chmod 600 ~/.kaggle/kaggle.json
```
4. Run **kaggle_dataset.py**
```
cd /FINAL-PROJECT-GROUP4/Code/
python3 kaggle_dataset.py
```

# <a name="data-preprocessing"></a>
## Data Preprocessing

#### Clean Text
1. Remove all text after: "TLDR", "TLDR:", "TL;DR:", "TL DR:", "TL DR".
2. Remove any links.
3. Remove '&amp', '&amp;#x200B;'.
4. Remove '***' or more.