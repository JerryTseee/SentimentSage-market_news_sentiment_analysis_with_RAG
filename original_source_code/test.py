import requests
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

"""
Author: Wangpok Tse
Email: crazytse@connect.hku.hk
7/24/2025, Hong Kong SAR
"""

# Set your API keys
OPENAI_API_KEY = ""
NEWS_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# STEP 1: Get financial news
def fetch_news(query, api_key=NEWS_API_KEY):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=10&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [f"{a['title']}. {a['description'] or ''}" for a in articles]


# STEP 2: FinBERT Sentiment
def get_finbert_pipeline():
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(articles, pipe):
    results = []
    for text in articles:
        sentiment = pipe(text[:512])[0]
        results.append({"text": text, "label": sentiment['label'], "score": sentiment['score']})
    return results


# STEP 3: RAG with LangChain
def build_rag_pipeline(articles):
    # Save to temporary text files
    with open("temp_articles.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(articles))

    loader = TextLoader("temp_articles.txt", encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    return qa_chain


# MAIN FUNCTION
if __name__ == "__main__":
    company = input("Enter the company ticker (e.g., AAPL): ").strip().upper()
    print(f"Fetching news about {company}...")
    raw_articles = fetch_news(company)

    print("Analyzing sentiment...")
    finbert = get_finbert_pipeline() # sentiment analysis pipeline
    sentiment_results = analyze_sentiment(raw_articles, finbert)

    df = pd.DataFrame(sentiment_results)
    print(df[["label", "score"]].value_counts())

    print("Generating summary via RAG...")
    qa = build_rag_pipeline([x["text"] for x in sentiment_results])
    result = qa.invoke(f"Can you provide a sentiment analysis and risk summary for {company} based on the news articles?")

    print("\nFinal Investment Summary:\n")
    print(result['result'])
