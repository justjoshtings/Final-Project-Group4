#!/bin/bash

echo "Executing Woby data sourcing, cleaning, and push to Kaggle."

echo "[Clearing data]..."
python3 clear_data.py

echo "[Scrapping Reddit]..."
python3 scrape_reddit.py

echo "[Post scrape data handling]..."
python3 post_scrape_reddit.py

echo "[Push dataset to Kaggle]..."
python3 kaggle_dataset_push.py