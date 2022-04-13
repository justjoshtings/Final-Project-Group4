"""
preprocess_corpus.py
Script to preprocess corpus data

author: @justjoshtings
created: 3/31/2022
"""
from Woby_Modules.CorpusProcessor import CorpusProcessor

def main():
    SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
    CORPUS_FILEPATH = '../corpus_data/'

    parser = CorpusProcessor(CORPUS_FILEPATH, log_file=SCRAPPER_LOG)

    print('Corpus size: ', parser.corpus_size()/1e6, 'MB')

    parser.clean_corpus()
    parser.EDA()
    parser.split_by_sentence()

if __name__ == "__main__":
    print("Executing preprocess_corpus.py")
    main()

