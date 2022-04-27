#!/bin/bash

echo "Installing MongoDB"
# Install and setup MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl status mongod
sudo systemctl enable mongod
# mongosh

echo "Executing Woby data sourcing, cleaning, and push to Kaggle."

echo "[Clearing data]..."
python3 clear_data.py

echo "[Scrapping Reddit]..."
python3 scrape_reddit.py

echo "[Post scrape data handling]..."
python3 post_scrape_reddit.py

echo "[Push dataset to Kaggle]..."
python3 kaggle_dataset_push.py