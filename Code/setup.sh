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

sudo apt install python3.8-venv
python3 -m venv ./myenv/
source myenv/bin/activate

# Install python requirements
pip3 install -r requirements.txt

# Install and setup MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl status mongod
sudo systemctl enable mongod
# mongosh
