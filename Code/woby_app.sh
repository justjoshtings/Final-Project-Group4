#!/bin/bash

echo "Executing Woby data download from Kaggle, perform EDA/processing, and modeling, and launch the Flask app."

# This is where user would run from
echo "[Pushing data to kaggle]..."
python3 kaggle_dataset.py

echo "[Preprocess text corpus and perform EDA]..."
python3 preprocess_corpus.py

# echo "[Performing modeling/finetuning]..."
# python3 modeling.py

echo "[Download Model Weights]..."
python3 download_model_weights.py

echo "[Model Evaluation]..."
# python3 download_model_weights.py

echo "[Launch Flask App]..."
python3 app.py