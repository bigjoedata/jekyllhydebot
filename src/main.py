import pandas as pd
import streamlit as st
import re
import random
import time
import streamlit.report_thread as ReportThread
from streamlit.server.server import Server
import SessionState
import torch
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from emoji_translate.emoji_translate import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Jekyll&Hyde ChatBot") #layout='wide', initial_sidebar_state='auto'

# set first td and first th of every table to not display
st.markdown("""
<style>
table td:nth-child(1) {
    display: none
}
table th:nth-child(1) {
    display: none
}
</style>
""", unsafe_allow_html=True)

def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:#3c403f  ; padding:15px">
    <h2 style = "color:white; text_align:center;"> {main_txt} </h2>
    <p style = "color:white; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def display_side_panel_header(txt):
    """
    function to display minor headers at side panel
    Parameters
    ----------
    txt: str -> the text to be displayed
    """
    st.sidebar.markdown(f'## {txt}')

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_emo():
    emo = Translator(exact_match_only=True, randomize=True)
    return emo

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_hyde():
    model = AutoModelForCausalLM.from_pretrained('bigjoedata/rockchatbot')
    return model

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_jekyll():
    model = AutoModelForCausalLM.from_pretrained('bigjoedata/friendlychatbot')
    return model

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small', TOKENIZERS_PARALLELISM=False)
    #Naughty word filtering not currently working
    #badfile = open("badwords.txt", "r")
    #badlist = []
    #badlist = [(line.strip()).split() for line in badfile]
    #badlist = list(filter(None, badlist))
    #badfile.close()
    #bad_word_ids = [tokenizer.encode(bad_word) for bad_word in badlist] # add_prefix_space=True
    return tokenizer#, bad_word_ids

@st.cache()
def cacherando():
    rando = random_number=random.random()
    return rando

@st.cache(max_entries=1)
def setstartpers(personalities, rando): #session_id):
    startpers=random.randint(0, 1)
    return startpers

def model_selector(personalities, startpers):
    personality = st.radio("Select A Personality: ", options=list(personalities), format_func=personalities.get, index=startpers)    #, key=personalities.get
    return personality

def get_session_state(rando):
    session_state = SessionState.get(sessionstep = 0, random_number=random.random(), chatinput='', personality='', personalityhistory=[],
                    bot_input_ids='', chat_history_ids='', chathistory=[], bothistory=[], emojitranslate='', length_gen='', temperature='',
                    topk='', topp='', no_repeat_ngram_size='')
    return session_state

def getanalyzer():
    return SentimentIntensityAnalyzer()

def mychat(): #, bad_line_ids): #emojitranslate, length_gen, temperature, topk, topp, no_repeat_ngram_size, model, session_state, tokenizer, emo, 
    rando = cacherando()
    session_state = get_session_state(rando)
    personalities = {
            "jekyll": "🧑‍🔬Dr. Jekyll🥰",
            "hyde": "👹Mr. Hyde👿"
        }
    startpers = setstartpers(personalities, rando)
    session_state.personality = model_selector(personalities, startpers)

    display_side_panel_header("Configuration & Fine-Tuning")

    session_state.emojitranslate = st.sidebar.checkbox("Emoji Translation? (Converts emojis to words)")
    # Max is currently 512 with this model but set lower here to reduce computational intensity

    session_state.length_gen = st.sidebar.select_slider(
        "Response Length (i.e., words/word-pairs) ", [r * 64 for r in range(3, 17)], 512)
    session_state.temperature = st.sidebar.slider("Choose temperature. Higher means more creative (crazier): ", 0.0, 3.0, 0.8, 0.1)
    session_state.topk = st.sidebar.slider("Choose Top K, the number of words considered at each step. Higher is more diverse; 0 means infinite:", 0, 200, 50)
    session_state.topp = st.sidebar.slider("Choose Top P. Limits next word choice to higher probability; lower allows more flexibility:", 0.0, 1.0, 0.95, 0.05)
    session_state.no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size. Eliminates repeated phrases of N Length", 0, 6, 2)

    with st.spinner("Initial models loading, please be patient"):
        tokenizer = load_tokenizer() #  bad_word_ids
        emo = load_emo()
        analyzer = getanalyzer()
        hyde = load_hyde()
        jekyll = load_jekyll()
        
    if session_state.personality == "hyde":
        model = hyde
    elif session_state.personality == "jekyll":
        model = jekyll

    chatlogholder = st.empty()
    chatholder = st.empty()

    keepers = "'_ ,-?!"
    chatinput = re.sub('([a-zA-Z])', lambda x: x.groups()[0].upper(),re.sub(
        r'[^\w+'+keepers+']', '', chatholder.text_input(">> You:", value="")), 1).strip() #, key=session_state.sessionstep
    try:
        if st.button("Chat"):
        #if len(chatinput) > 1:
            session_state.sessionstep += 1
            # session_state.sessionstep # uncomment to see if conversation is progressing
            chatinput = chatinput
            
            new_user_input_ids = tokenizer.encode(chatinput + tokenizer.eos_token, return_tensors='pt')
            if session_state.sessionstep == 1:
                bot_input_ids = new_user_input_ids
            elif session_state.sessionstep < 4:
                bot_input_ids = torch.cat([session_state.chat_history_ids, new_user_input_ids], dim=-1)
            else: # The model was only trained on a max of 6 interactions so we will slice the first two off.
                eostok = tokenizer.encode(tokenizer.eos_token, return_tensors='pt')
                firstchat = torch.nonzero(session_state.chat_history_ids == eostok)[1][1]
                removefirstchat = session_state.chat_history_ids[:,firstchat + 1:]
                bot_input_ids = torch.cat([removefirstchat, new_user_input_ids], dim=-1)

            session_state.chat_history_ids = model.generate(
                bot_input_ids, max_length=512,
                pad_token_id=tokenizer.eos_token_id,  
                no_repeat_ngram_size=session_state.no_repeat_ngram_size,       
                do_sample=True,
                top_k=session_state.topk, 
                top_p=session_state.topp,
                temperature = session_state.temperature,
                #bad_words_ids = bad_word_ids # bad word filtering not currently working
                #num_beams = 3,
                #num_beam_groups = 3
            )

            thechat = "{}".format(tokenizer.decode(session_state.chat_history_ids[:,
                                        bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

            session_state.personalityhistory.append(personalities.get(session_state.personality))
            session_state.chathistory.append("" + chatinput)

            chat_sentiment = analyzer.polarity_scores(thechat)['compound']

            if session_state.emojitranslate:
                thechat=emo.emojify(thechat)
            else:
                pass

            if chat_sentiment >= .7:
                thechat = emo.add_positive_emojis(thechat, num=2)
            elif .2 <= chat_sentiment < .7:
                thechat = emo.add_positive_emojis(thechat, num=1)
            elif -.2 <= chat_sentiment < .2:
                thechat = emo.add_neutral_emojis(thechat, num=1)
            elif -.7 <= chat_sentiment < -.2:
                thechat = emo.add_negative_emojis(thechat, num=1)
            else:
                thechat = emo.add_negative_emojis(thechat, num=2)

            session_state.bothistory.append(thechat)

            df = pd.DataFrame(
                {'Personality': session_state.personalityhistory,
                'You': session_state.chathistory,
                'Bot': session_state.bothistory
                })

            with chatlogholder:
                st.table(df.tail(5))
            with st.beta_expander("Full chat history"):
                st.table(df)       
    except:
        raise


def main():
    main_txt = """💬🧑‍🔬👹 Welcome To Jekyll&Hyde ChatBot  🥰👿🤘"""
    sub_txt = "Just have fun"
    subtitle = """
A chatbot with two distinct personalities, you may switch between anytime. **[Dr. Jekyll](https://huggingface.co/bigjoedata/friendlychatbot)** is trained on a selected portions of movie & tv scripts, Facebook [Empathetic Dialogs](https://github.com/facebookresearch/EmpatheticDialogues) and various dialog scattered across the web. **[Mr. Hyde](https://huggingface.co/bigjoedata/rockchatbot)** is trained on the writing styles of nearly 20k songs by over 400 artists across MANY musical genres (not just rock), as well as a few poets and comedians.

Due to the imprecise nature of the training materials, the results can be unpredictable. I have made no attempt to censor the source materials. Dr. Jekyll tends to be nicer while Mr. Hyde tends to be ruder and unpredictable. Either of them might use NSFW language. The bot will also add emojis to emphasize their feelings. Just have fun!

**Instructions:** Type in some text and click "Chat" to generate a response. Optionally, adjust settings on the left.
        """

    display_app_header(main_txt,sub_txt,is_sidebar = False)
    st.markdown(subtitle)
    display_side_panel_header("Rockbot!")
    st.sidebar.markdown("""
                        [Github](https://github.com/bigjoedata/rockbot)  
                        [Primary Model](https://huggingface.co/bigjoedata/rockbot)  
                        [Distilled Model](https://huggingface.co/bigjoedata/rockbot-distilgpt2/)""")
    mychat()
    st.markdown("""
---
💬🧑‍🔬👹 🥰👿🤘
## Dr. Jekyll & Mr. Hyde Chatbot Background
I trained a [lyrics generating tool](https://github.com/bigjoedata/rockbot) on the same music lyrics Mr. Hyde was trained on ([demo available here](https://share.streamlit.io/bigjoedata/rockbot/main/src/main.py)) and thought it would be fun to build a chatbot to interpret the lyrics in a more abstract way. I looked and found some similar projects and trained my model in a similar way, but thought a counterpart personality would be fun to create as well.

A demo is available [here](https://share.streamlit.io/bigjoedata/rockchatbot/main/src/main.py) Check out [Github](https://github.com/bigjoedata/rockbot) to spin up your own Rockbot.

- Model creation was done by fine tuning [DialoGPT](https://github.com/microsoft/DialoGPT), An OpenAI [GPT-2](https://github.com/openai/gpt-2) and [Huggingface Transformer](https://github.com/huggingface/transfer-learning-conv-ai) based model.
-  [Dr. Jekyll Model](https://huggingface.co/bigjoedata/friendlychatbot)
 - [Mr. Hyde Model](https://huggingface.co/bigjoedata/rockchatbot) 
 - The UI / back end is built in [Streamlit](https://www.streamlit.io/)
 - [GPT-2 generation](https://huggingface.co/blog/how-to-generate)
 - [Illustrated Transformers Models](http://jalammar.github.io/illustrated-transformer/)
 - [LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/)   (retrieving lyrics for training).
 - [Knime](https://www.knime.com/) (data cleaning and post processing)
 - [Emoji Translator](https://github.com/fabriceyhc/emoji_translate) Used to convert text to emojis and add emoji suffixes to bot
 - [Vader](https://github.com/cjhutto/vaderSentiment) Used for sentiment analysis 

**Similar projects**
 - [Open-Dialog Chatbots for Learning New Languages](https://nathancooper.io/i-am-a-nerd/chatbot/deep-learning/gpt2/2020/05/12/chatbot-part-1.html)
 - [Rick and Morty Chatbot](https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30)

**Mr. Hyde Data Prep Cleaning Notes:**
- Removed duplicate lyrics from each song
- Deduped similar songs based on overall similarity to remove cover versions
- Removed as much noise / junk as possible. There is still some.
- Added tokens to delineate song
- Used language to remove non-English versions of songs
- Many others!

## How to Use The Model
Please refer to [Huggingface DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium)
 
## Spin up your own with Docker
Running your own is very easy. Visit my [Streamlit-Plus repository](https://github.com/bigjoedata/streamlit-plus) for more details on the image build

 - Install [Docker Compose](https://docs.docker.com/compose/install/)
 - Follow the following steps
```
git clone https://github.com/bigjoedata/jekyllhydebot
cd jekyllhydebot
nano docker-compose.yml # Edit as needed
docker-compose up -d # launch in daemon (background) mode
```
""")

if __name__ == "__main__":
    main()