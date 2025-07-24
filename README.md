<img width="300" height="300" alt="logo" src="https://github.com/user-attachments/assets/9a79ea9c-9cc5-45ca-ac7f-469757921619" />

[GitHub](https://github.com/JerryTseee/SentimentSage-market_news_sentiment_analysis_with_RAG.git)👀    [Official Website](https://sentimentsage-marketnewssentimentanalysiswithrag-nrgktn4pk5dvw.streamlit.app/)✨
  
# SentimentSage - Market News Sentiment Analyzer 😎
SentimentSage is a powerful AI-driven platform designed to provide deep insights and analytics on financial news and market sentiment. By leveraging cutting-edge Natural Language Processing (NLP) models like FinBERT, combined with advanced Retrieval-Augmented Generation (RAG) technology, SentimentSage can quickly analyze vast amounts of financial data, extract sentiment, and generate accurate, actionable investment summaries.

- original source code: only includes python code and run it on terminal
- software: web-based UI

## Original Source Code 📦
- register Openai API key and NewsAPI Key
- replace the api keys on paths in the test.py
- pip install -r requirements.txt
- run the test.py
- then you can use the analyze tool, start with input your chosen company ticker

## Software Version 🚀
![1](https://github.com/user-attachments/assets/eec3bc8b-9b3e-4aaa-ac0d-66070c9c983e)
  
![2](https://github.com/user-attachments/assets/db02261f-154a-4b5a-9e24-ac9380755a46)
  
![3](https://github.com/user-attachments/assets/8a77d1c9-31a9-485f-aa22-fe4c80e012dd)

to deploy the application:
- Streamlit Cloud
- Create a new repository on GitHub
- Upload your files
- Make sure your repository is public
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file path: `streamlit_app.py`
- Click "Deploy"
- Add secrets in the "Secrets" section:
- OPENAI_API_KEY = "your-openai-key"
- NEWS_API_KEY = "your-news-api-key"
