#!/bin/bash

echo "Project Setup for Woby Tales"
echo "Use Ubuntu 20"

# Ensure python3 and pip are installed
sudo apt update
sudo apt install -y python3-pip

# Clone project
cd ~/
git clone https://github.com/justjoshtings/Final-Project-Group4.git

cd ./Final-Project-Group4

sudo apt -y install python3.8-venv
python3 -m venv ./myenv/
source myenv/bin/activate

# Install python requirements
pip3 install -r requirements.txt

# Set up Kaggle API
echo "Setup Kaggle API and download kaggle.json"

FILE=~/.kaggle/kaggle.json
echo "Checking if kaggle.json exists in: $FILE"

if test -f "$FILE"; then
    echo "$FILE exists."
    chmod 600 ~/.kaggle/kaggle.json

    echo "Testing kaggle API, running 'kaggle competitions list'"
    kaggle competitions list
else 
    echo "Set up Kaggle API with the following resources and download kaggle.json to ~/.kaggle/kaggle.json"
    echo "https://adityashrm21.github.io/Setting-Up-Kaggle/"
    echo "https://github.com/Kaggle/kaggle-api#api-credentials"
    echo "Or ignore and download manually instead. Check README data-download section for more."
fi

# Make Log File
mkdir ./Woby_Log/
cd ./Woby_Log/
touch ScrapperLog.log
cd ../Code/
