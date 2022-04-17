#import files
from flask import Flask, render_template, request
# from waitress import serve
from Woby_Modules.LanguageModel import LanguageModel_GPT2, LanguageModel_GPT_NEO, CustomTextDatasetGPT2
from transformers import Conversation, GPT2Tokenizer

SCRAPPER_LOG = '../Woby_Log/ScrapperLog.log'
CORPUS_FILEPATH = '../corpus_data/'

random_state = 42

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

model_gpt2 = LanguageModel_GPT2(corpus_filepath=CORPUS_FILEPATH, 
								random_state=random_state, 
								train_data_loader=None,
								valid_data_loader=None,
								test_data_loader=None,
								gpt_model_type='gpt2',
								log_file=SCRAPPER_LOG)

model_gpt2.load_weights('./results/model_weights/gpt2_10epochs_finetuned/')

global chats
chats = list()

app = Flask(__name__)

@app.route("/")
def home():    
    return render_template("home.html") 

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    try:
        bot_Text = str(model_gpt2.generate_text_for_web(chats[-1]+userText))
    except IndexError:
        bot_Text = str(model_gpt2.generate_text_for_web(userText))
    chats.append(userText)
    chats.append(bot_Text)
    return bot_Text

if __name__ == "__main__":
    print('Running Woby App')
    port = 8080
    # serve(app, host="0.0.0.0", port=someport, threads=8)
    app.run(port = port, debug = True, threaded = True)

