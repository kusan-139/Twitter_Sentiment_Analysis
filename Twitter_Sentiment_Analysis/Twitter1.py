import os
import sys
import logging
import re
import string
import contextlib
import subprocess
import time
from dotenv import load_dotenv

packages = [
    "transformers",
    "nltk",
    "python-dotenv",
    "tweepy",
    "pandas",
    "textblob",
    "vaderSentiment",
    "torch", # Added torch as it's a dependency for transformers
]

print("Checking and installing required packages...")
for pkg in packages:
    try:
        # Check if the package is importable
        __import__(pkg.split('-')[0])
    except ImportError:
        print(f"Installing {pkg}...")
        try:
            # Install the package using pip
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {pkg}. Please install it manually.", file=sys.stderr)
            sys.exit(1)
print("All packages are ready.")

# Now that packages are installed, we can import them
import tweepy
from transformers import pipeline
import nltk
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Suppress TensorFlow and other verbose logging ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Base directory and .env path configuration ---
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

# --- NLTK resource download ---
# This is done silently to avoid cluttering the output.
with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
from nltk.tokenize import sent_tokenize

# --- Constants and Mappings ---
EMOJI_MAP = {"positive": "😄", "negative": "😞", "neutral": "😐"}

# Keywords for heuristic-based sentiment adjustments
POS_KEYWORDS = [
    "rescue", "recovered", "win", "success", "achieve", "growth", "profit",
    "improve", "graduate", "graduated", "first-class", "honours", "award",
    "scholarship", "excellent", "passed", "1 million views", "Guaranteed A", "Grade A"
]
NEG_KEYWORDS = ["kill", "dead", "death", "fraud", "crash", "attack", "loss", "disaster", "error", "slower"]
EMOJI_KEYWORDS = {
    "😄": "positive", "😊": "positive", "👍": "positive", "🏆": "positive", "💰": "positive",
    "😞": "negative", "😡": "negative", "👎": "negative", "💸": "positive", "⚠️": "negative",
    "😐": "neutral", "😶": "neutral"
}

# --- Twitter API Configuration ---
BEARER_TOKEN = os.getenv("BEARER_TOKEN1")

def init_twitter_client():
    """Initializes and returns a Tweepy client."""
    if not BEARER_TOKEN:
        print("Error: No bearer token (BEARER_TOKEN1) found in .env file.", file=sys.stderr)
        return None
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=False)
        # Test the client connection with a minimal query
        client.search_recent_tweets(query="test", max_results=10)
        return client
    except tweepy.errors.TooManyRequests:
        print("Error: Rate limit exceeded for your bearer token. Please try again after 15 minutes.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: Failed to initialize Twitter client. Details: {e}", file=sys.stderr)
        return None

# --- Initialize Sentiment Analysis Models ---
def get_pipelines_and_analyzer():
    """Loads and returns all sentiment analysis models."""
    print("Loading sentiment analysis models...")
    try:
        twitter_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        vader_analyzer = SentimentIntensityAnalyzer()
        print("Models loaded successfully.")
        return twitter_pipeline, finbert_pipeline, vader_analyzer
    except Exception as e:
        print(f"Error loading sentiment models: {e}", file=sys.stderr)
        print("Please ensure you have a stable internet connection.", file=sys.stderr)
        return None, None, None

# --- Utility Functions ---
def normalize_text(text):
    """Lowercase and remove punctuation."""
    return ''.join(c for c in text.lower() if c not in string.punctuation).strip()

def apply_heuristics(text, label):
    """Adjusts sentiment label based on positive/negative keywords."""
    t = normalize_text(text)
    if any(word.lower() in t for word in POS_KEYWORDS):
        return "positive"
    if any(word.lower() in t for word in NEG_KEYWORDS):
        return "negative"
    return label

def apply_emoji_heuristics(text, label):
    """Adjusts sentiment label based on emojis present in the text."""
    for emoji, sentiment in EMOJI_KEYWORDS.items():
        if emoji in text:
            return sentiment
    return label

def is_financial_tweet(text):
    """Checks if a tweet contains financial keywords."""
    finance_keywords = [
        "stock", "market", "shares", "profit", "loss", "trade", "bank", "economy",
        "inflation", "investor", "fund", "ipo", "finance", "merger", "deal"
    ]
    return any(word in text.lower() for word in finance_keywords)

def clean_tweet(text):
    """Removes RT, mentions, URLs, and hashtags from a tweet."""
    text = re.sub(r'RT @\w+: ', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text.strip()

# --- Core Functions ---
def fetch_tweets(client, query, max_results=20):
    """Fetches and cleans recent tweets for a given query."""
    if not client:
        return []
    
    print(f"Fetching up to {max_results} tweets for '{query}'...")
    seen = set()
    tweets_list = []
    try:
        resp = client.search_recent_tweets(query=f"{query} -is:retweet lang:en", max_results=max_results)
        if resp.data is None:
            return []
        for tweet in resp.data:
            cleaned = clean_tweet(tweet.text)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                tweets_list.append(cleaned)
        return tweets_list
    except tweepy.errors.TooManyRequests:
        print("Warning: Rate limit exceeded. Please try again in 15 minutes.", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching tweets: {e}", file=sys.stderr)
        return []

def analyze_sentiment(tweet_texts, twitter_pipeline, finbert_pipeline, vader_analyzer):
    """Analyzes sentiment for a list of tweets using an ensemble of models."""
    tweet_sentiments = []
    
    for tweet in tweet_texts:
        sentences = sent_tokenize(tweet)
        scores = {"positive": 0, "negative": 0, "neutral": 0}

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence_scores = {"positive": 0, "negative": 0, "neutral": 0}

            # 1. BERT models (RoBERTa or FinBERT) - Weight: 0.5
            pipeline_to_use = finbert_pipeline if is_financial_tweet(sentence) else twitter_pipeline
            result = pipeline_to_use(sentence[:512])[0]
            label = result['label'].lower()
            if result['score'] < 0.6:
                label = "neutral"
            
            # Apply keyword and emoji heuristics
            label = apply_heuristics(sentence, label)
            label = apply_emoji_heuristics(sentence, label)
            sentence_scores[label] += 0.5

            # 2. TextBlob - Weight: 0.2
            polarity = TextBlob(sentence).sentiment.polarity
            tb_label = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
            sentence_scores[tb_label] += 0.2

            # 3. VADER - Weight: 0.3
            vader_scores = vader_analyzer.polarity_scores(sentence)
            vader_label = "positive" if vader_scores['compound'] > 0.05 else "negative" if vader_scores['compound'] < -0.05 else "neutral"
            sentence_scores[vader_label] += 0.3

            # Aggregate scores for the sentence
            for k in scores:
                scores[k] += sentence_scores[k]

        # Determine overall sentiment for the tweet
        overall_label = max(scores, key=scores.get)
        tweet_sentiments.append((overall_label, tweet))

    return tweet_sentiments

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Parameters ---
    SEARCH_QUERY = "NASA"
    NUM_TWEETS = 25

    # --- Initialization ---
    client = init_twitter_client()
    twitter_pipe, finbert_pipe, vader_analyzer = get_pipelines_and_analyzer()

    if not client or not twitter_pipe:
        print("\nExiting due to initialization failure.")
        sys.exit(1)

    # --- Fetch and Analyze ---
    tweets = fetch_tweets(client, SEARCH_QUERY, NUM_TWEETS)

    if not tweets:
        print(f"\nNo tweets found for the query: '{SEARCH_QUERY}'")
    else:
        print(f"Found {len(tweets)} unique tweets. Analyzing sentiment...")
        tweet_sentiments = analyze_sentiment(tweets, twitter_pipe, finbert_pipe, vader_analyzer)

        # --- Display Results ---
        print("\n--- Sentiment Analysis Results ---")
        
        # 1. Summary
        sentiment_counts = pd.DataFrame(tweet_sentiments, columns=['label', 'text'])['label'].value_counts()
        print("\nSentiment Distribution:")
        # Iterate through the sentiment counts and print with emoji
        for sentiment, count in sentiment_counts.items():
            emoji = EMOJI_MAP.get(sentiment, "❓") # Get emoji from map, with a fallback
            print(f"{emoji} {count}")
        
        # 2. Detailed List
        print("\n--- Tweets & Sentiments ---")
        for label, tweet in tweet_sentiments:
            print(f"{EMOJI_MAP[label]} [{label.upper()}] {tweet}")

