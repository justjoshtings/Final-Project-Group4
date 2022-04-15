"""
download_model_weights.py
Script to download Github release .zip files

author: @justjoshtings
created: 4/14/2022
"""
import requests, zipfile, io, os

def main():
    model_weights_zip_url = 'https://github.com/justjoshtings/Final-Project-Group4/archive/refs/tags/v0.1.0-alpha.zip'

    r = requests.get(model_weights_zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    if not os.path.exists('../results_test/model_weights/'):
        os.makedirs('../results_test/model_weights/')

    z.extractall("../results_test/model_weights/")

    # make folder for gpt2_final_weights, gptneo_final_weights and download each one individually into it


if __name__ == "__main__":
    print("Executing download_model_weights.py")
    main()