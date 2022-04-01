## To Do
#### Preprocessing
* store on google 
* download from google
* calc data size of google download
* Review some data
* clean data
* EDA

#### Model Pipeline
* figure out pipeline for gpt2, gpt neo, bert, custom model
* What inputs are needed for each model?

#### Model Training/Finetuning

#### Model Evaluation

## Contents
1. sample_results
2. Woby_keys
3. Woby_Modules
4.

## Data Acquisition

Data Acquisition from posts of different sub-reddits were done with two primary services:

1. [Reddit Dev API](https://www.reddit.com/dev/api/) with **requests**
2. [Python Pushshift.io API Wrapper (PSAW)](https://psaw.readthedocs.io/en/latest/)

The **Reddit Dev API** has a limit of 1000 Reddit post submissions it can GET. **PSAW** theorectically has all historical post submissions but it has issues where some shards may be out of service and therefore not all post submissions can be retrieved. I used a combination of both services then removed duplciates in order to get the largest corpus possible. 

The **Reddit Dev API** and **PSAW** returned a total of 19,727 posts from the subreddits. This final number becomes 13,120 after removing duplicated stories and 137MBs.

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
|─── Corpus
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