#!/bin/bash

echo "Executing Woby data download from Kaggle, perform EDA/processing, and modeling, and launch the Flask app."

# This is where user would run from
echo "[Pulling data from Kaggle]..."
python3 kaggle_dataset_down.py

echo "[Preprocess text corpus and perform EDA]..."
python3 preprocess_corpus.py

echo "[Modeling Finetuning]..."
echo "If needed, please run 'python3 gpt2spooky_pretraining.py' then 'modeling.py'..."
# python3 gpt2spooky_pretraining.py
# python3 modeling.py

echo "[Download Model Weights]..."
python3 download_model_weights.py

echo "[Model Evaluation]..."
python3 model_evaluation.py

echo "[Launch Flask App]..."
python3 app.py