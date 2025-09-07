**Twitter Sentiment Analysis** 

This is a Python script that fetches recent tweets based on a user-defined search query and performs sentiment analysis on them. It uses an ensemble of powerful NLP models to classify the sentiment of each tweet as positive, negative, or neutral and then displays the results directly in your terminal.

**Features**

Real-time Tweet Fetching: Utilizes the Twitter v2 API to fetch the most recent tweets.

Ensemble Sentiment Analysis: Combines multiple models for more accurate analysis:

twitter-roberta-base-sentiment-latest: A robust model fine-tuned on Twitter data.

FinBERT: A specialized model for financial-related tweets.

TextBlob: A classic rule-based sentiment analyzer.

VADER: A lexicon and rule-based tool specifically attuned to sentiments expressed in social media.

Heuristic Adjustments: Uses keyword and emoji heuristics to improve sentiment classification accuracy.

Automatic Dependency Installation: Checks for and installs required Python packages on the first run.

Simple Command-Line Interface: Easy to run and view results directly in the terminal.

Requirements

Python 3.7+

A Twitter Developer account with a Bearer Token for the v2 API.

Setup and Usage
Follow these steps to get the tool up and running.

1. Download the Script
Save the Python script as Twitter1.py in a new folder.

2. Create the Environment File

In the same folder as the script, open a file named .env. This file will store your Twitter API bearer token securely.

Open the .env file and add your bearer token like this:

BEARER_TOKEN1="YOUR_TWITTER_BEARER_TOKEN_HERE"

Replace "YOUR_TWITTER_BEARER_TOKEN_HERE" with the actual bearer token from your Twitter Developer dashboard.

3. Run the Script

Open your terminal or command prompt, navigate to the folder where you saved the files, and run the script using Python:

python Twitter1.py

The first time you run it, the script will automatically install all the necessary packages like tweepy, transformers, torch, etc. This might take a few minutes. Subsequent runs will be much faster.

How It Works
Initialization: The script loads your bearer token from the .env file and initializes the Tweepy client to connect to the Twitter API.

Model Loading: It loads the pre-trained sentiment analysis models into memory.

Tweet Fetching: It fetches the most recent, non-retweet English tweets matching the SEARCH_QUERY.

Sentiment Analysis: For each tweet, it performs the following:

Splits the tweet into sentences.

Analyzes each sentence using a weighted combination of the four sentiment models.

Applies keyword and emoji heuristics to refine the sentiment score.

Aggregates the sentence scores to determine the overall sentiment for the tweet.

Display Results: It prints a summary of the sentiment distribution (how many positive, negative, and neutral tweets were found) followed by a detailed list of each tweet and its classified sentiment.

Customization
You can easily change the search query and the number of tweets to analyze by editing the __main__ block at the bottom of the script:

if __name__ == "__main__":
    # --- Parameters ---
    # Change this to any topic you want to analyze
    SEARCH_QUERY = "NASA"
    # Change this to the number of tweets you want to fetch (max 100)
    NUM_TWEETS = 50

    # ... rest of the script

SEARCH_QUERY: The topic you want to search for on Twitter.

NUM_TWEETS: The number of recent tweets to fetch for analysis (the Twitter API limit is 100 for recent searches).


**Note** : Please install the "requirements.txt" for all library and packages.


