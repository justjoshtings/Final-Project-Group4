#!/bin/bash

# echo "Executing Woby data sourcing, cleaning, and push to Kaggle."

# echo "[Clearing data]..."
# python3 clear_data.py

# echo "[Scrapping Reddit]..."
# python3 scrape_reddit.py

# echo "[Post scrape data handling]..."
# python3 post_scrape_reddit.py

# This is where user would run from
echo "[Pushing data to kaggle]..."
python3 kaggle_dataset.py

echo "[Preprocess text corpus and perform EDA]..."
python3 preprocess_corpus.py

echo "[Modeling Test]..."
python3 modeling.py