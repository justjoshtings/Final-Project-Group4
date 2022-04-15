"""
download_model_weights.py
Script to download Github release .zip files

author: @justjoshtings
created: 4/14/2022
"""
import requests, zipfile, io, os, shutil

def main():
    zip_folder('./results/model_weights/gpt2', './results/model_weights/gpt2_model_finetuned')
    zip_folder('./results/model_weights/gpt_neo_125M', './results/model_weights/gpt_neo_125M_model_finetuned')

    download_and_unzip('https://github.com/justjoshtings/Final-Project-Group4/releases/download/v0.1.1-alpha/gpt2_model_finetuned.zip', './results/model_weights/gpt2_finetuned')
    download_and_unzip('https://github.com/justjoshtings/Final-Project-Group4/releases/download/v0.1.1-alpha/gpt_neo_125M_model_finetuned.zip', './results/model_weights/gpt2_neo_finetuned')

def zip_folder(dir_name, output_file_name):
    '''
    Function to zip folder

    Params:
        dir_name (str): path to folder
        output_file_name (str): name of output zip file
    '''
    try:
        shutil.make_archive(output_file_name, 'zip', dir_name)
    except FileNotFoundError:
        print(f'{dir_name} not found!')
        pass
    
def download_and_unzip(zip_url, output_dir_name):
    '''
    Function to download zip url and unzip

    Params:
        zip_url (str): zip url
        output_dir_name (str): name of output directory
    '''
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    z.extractall(output_dir_name)


if __name__ == "__main__":
    print("Executing download_model_weights.py")
    main()