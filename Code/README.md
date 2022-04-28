# Code

## Description
This directory holds all relevant code for data acquisition, preprocessing, model building/training, evaluation, and front end.

# <a name="app-execution"></a>
## App Execution

After cloning the repo, navigate to the Code folder and set permissions for 4 bash scripts.
```
cd Final-Project-Group4/Code/
chmod u+x env_setup.sh
chmod u+x woby_app.sh
chmod u+x woby_app_manual_kaggle.sh
chmod u+x data_setup.sh
```
Below is the description of each script:
1. env_setup.sh - runs environment setup and installs needed software
2. woby_app.sh - pulls data from Kaggle, downloads finetuned model weights, preprocesses data, evaluate models, launches Flask app with best model (model pretrainig and finetuning are skipped and downloaded weights are used instead)
3. data_setup.sh - set up MongoDB database, pull data from Reddit APIs, and pushes data into Kaggle (DO NOT NEED TO RUN)

Next, you can either download data from Kaggle manually or setup Kaggle API credentials to download through a prepared script. See [data download](https://github.com/justjoshtings/Final-Project-Group4/blob/main/Code/README.md#data-download) section for more details on both options.

Next, run the env_setup.sh script.
```
cd Final-Project-Group4/Code/
./env_setup.sh
```

Next, activate virtual environment
```
cd ../
source myenv/bin/activate
cd ./Code/
```

Next, run woby_app.sh if you set up Kaggle API credentials. If not, run woby_app_manual_kaggle.sh instead.
```
./woby_app.sh
```
or
```
./woby_app_manual_kaggle.sh
```
Next, open a browser and navigate to **http://127.0.0.1:8080** in order to see the Woby Flask App.

## Contents
1. **results**: directory to save results about corpus, model weights, plots, and other results.
2. **sample_results**: sample json returns from Reddit PSAW and PRAW APIs
3. **templates**: html templates for Woby Flask Front End
2. **Woby_keys**: save credentials here
3. **Woby_Modules**: core custom python modules for Woby 
4. **app.py**: to start Flask app
5. **clear_data.py**: script to clear corpus data from MongoDB and locally saved .txt files
6. **data_setup.sh**: bash script to run set up MongoDB database, pull data from Reddit APIs, and pushes data into Kaggle 
7. **download_model_weights.py**: script to download Github release .zip files
8. **env_setup.sh**: bash script to perform environment and project setup for machine
9. **gpt2spooky_pretraining.py**: script to perform gpt2spooky pretraining on our custom corpus
10. **kaggle_dataset_down.py**: script to download Kaggle datasets
11. **kaggle_dataset_push.py**: script to upload Kaggle datasets
12. **model_evaluation.py**: script to perform modeling evaluation
13. **modeling.py**: script to perform modeling
14. **post_scrape_reddit.py**: script to run post scrape_reddit.py to remove duplicated stories and create .csv of metadata to save in /corpus
15. **preprocess_corpus.py**: script to preprocess corpus data
16. **scrape_reddit.py**: script to scrape Reddit to get corpus data and then save to disk/MongoDB
17. **test_RedditAPI.py**: testing script of RedditAPI module
18. **test.py**: script to perform some ad-hoc testing of models, not used for application core
19. **woby_app_manual_kaggle.sh**: bash script to perform EDA/processing, and modeling, and launch the Flask app. Skipping Kaggle download.
20. **woby_app.sh**: bash script to perform EDA/processing, and modeling, and launch the Flask app.

# <a name="data-acquisition"></a>
## Data Acquisition

Data Acquisition from posts of different sub-reddits were done with two primary services:

1. [Reddit Dev API](https://www.reddit.com/dev/api/) with **requests**
2. [Python Pushshift.io API Wrapper (PSAW)](https://psaw.readthedocs.io/en/latest/)

The **Reddit Dev API** has a limit of 1000 Reddit post submissions it can GET. **PSAW** theorectically has all historical post submissions but it has issues where some shards may be out of service and therefore not all post submissions can be retrieved. I used a combination of both services then removed duplciates in order to get the largest corpus possible. 

The **Reddit Dev API** and **PSAW** returned a total of 23,350 posts from the subreddits. This final number becomes 14,715 after removing duplicated stories and 137MBs.

For each post retrieved, the JSON response was parsed for metadata which was inserted into a MongoDB database while a copy of the submission text was saved into the **corpus** directory organized by sub-reddit with a [doc_id]_[t3]_[reddit_post_id].txt schema.

[MongoDB Setup](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/)

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
1. Manually download, unzip, and move all contents within **archive** folder to **FINAL-PROJECT-GROUP4/corpus_data/**. You will need to create the corpus_data directory. You can use scp to move files from local machine to a remote machine if needed.

**Option 2:** Use Kaggle API to download data
1. Make .kaggle directory
```
mkdir ~/.kaggle/
```
2. Create a Kaggle account API. See [here](https://github.com/Kaggle/kaggle-api#api-credentials) or [here](https://adityashrm21.github.io/Setting-Up-Kaggle/).
3. Download the kaggle.json file of your API credentials and save to **~/.kaggle/kaggle.json**
```
mv [downloaded kaggle.json path] ~/.kaggle/kaggle.json
```
4. Set permissions.
```
chmod 600 ~/.kaggle/kaggle.json
```

# <a name="data-preprocessing"></a>
## Data Preprocessing

#### Clean Text
1. Remove all text after: "TLDR", "TLDR:", "TL;DR:", "TL DR:", "TL DR".
2. Remove any links.
3. Remove '&amp', '&amp;#x200B;'.
4. Remove '***' or more.


## References
1. Huggingface Transformer Finetuning: https://huggingface.co/docs/transformers/training#finetune-in-native-pytorch
2. Huggingface How to Text Generation: https://huggingface.co/blog/how-to-generate
3. Huggingface Text Generation: https://huggingface.co/docs/transformers/main_classes/model#transformers.generation_utils.GenerationMixin
4. Fine tune GPT2: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=gpt6tR83keZD
5. BART: https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html
6. Flask Chatbot: https://dev.to/sahilrajput/build-a-chatbot-using-flask-in-5-minutes-574i

#### TO DO
* Final test of running from new ec2 [Wed]
      - test manual
      - test kaggle

* Final report [Thurs/Fri/Sat]
* Presentation [Sat/Sun/Mon]
* test on micro instance