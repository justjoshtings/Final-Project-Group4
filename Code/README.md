#### TO DO
* test on EC2 #2 env_setup.sh and woby_app.sh [Tues]

* review evaluation table and choose best model + push .xlsx file to here + make a plot of results + do hypothesis testing on 3 [Wed]
* setup Flask app with optimal model [Wed]
* cleanup code base and github [Wed]
* type out all readmes [Wed]
      - Code
            - results
            - sample_results
            - templates
            - Woby_modules
      - Group-Proposal
      - Final-Report

* Final report [Thurs/Fri/Sat]
* Final test of running from new ec2 [Sat]
* Presentation [Sun/Mon]

* final report word doc
* presentation
* github readmes

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
./env_setup.sh
```
Next, run woby_app.sh if you set up Kaggle API credentials. If not, run woby_app_manual_kaggle.sh instead.
```
./woby_app.sh
```
or
```
./woby_app_manual_kaggle.sh
```
Next, open a browser and navigate to **http://[your machine's public IP address]:8080** in order to see the Woby Flask App.

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

The **Reddit Dev API** and **PSAW** returned a total of 23,350 posts from the subreddits. This final number becomes 14,715 after removing duplicated stories and 137MBs.

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
1. Manually download, unzip, and move all contents within **archive** folder to **FINAL-PROJECT-GROUP4/corpus_data/**. You will need to create the corpus_data directory. You can use scp to move files from local machine to a remote machine if needed.

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

# <a name="data-preprocessing"></a>
## Data Preprocessing

#### Clean Text
1. Remove all text after: "TLDR", "TLDR:", "TL;DR:", "TL DR:", "TL DR".
2. Remove any links.
3. Remove '&amp', '&amp;#x200B;'.
4. Remove '***' or more.


## References
1. https://huggingface.co/docs/transformers/training#finetune-in-native-pytorch
2. https://huggingface.co/blog/how-to-generate
3. https://huggingface.co/docs/transformers/main_classes/model#transformers.generation_utils.GenerationMixin
4. Fine tune GPT2: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=gpt6tR83keZD
5. BART: https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html