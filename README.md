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
