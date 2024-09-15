import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pytube import YouTube
import requests
from transformers import pipeline
from datetime import datetime, timedelta

# Initialize the sentiment analysis pipeline
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Expanded stock dictionary with names and symbols
stock_dict = {
    'AAPL': 'Apple',
    'TSLA': 'Tesla',
    'GOOGL': 'Google',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'META': 'Meta Platforms',
    'NFLX': 'Netflix',
    'NVDA': 'NVIDIA',
    'BRK-B': 'Berkshire Hathaway',
    'JPM': 'JPMorgan Chase',
    'V': 'Visa',
    'MA': 'Mastercard',
    'DIS': 'Walt Disney',
    'BA': 'Boeing',
    'HD': 'Home Depot',
    'IBM': 'IBM',
    'PFE': 'Pfizer',
    'CSCO': 'Cisco Systems',
    'ADBE': 'Adobe',
    'INTC': 'Intel',
    'ORCL': 'Oracle',
    'COST': 'Costco',
    'WMT': 'Walmart',
    'T': 'AT&T',
    'KO': 'Coca-Cola',
    'XOM': 'ExxonMobil',
    'CVX': 'Chevron',
    'LMT': 'Lockheed Martin',
    'MCD': 'McDonald\'s',
    'NKE': 'Nike',
    'UNH': 'UnitedHealth Group',
    'MDT': 'Medtronic',
    'GILD': 'Gilead Sciences',
    'MRK': 'Merck',
    'ABBV': 'AbbVie',
    'BMY': 'Bristol-Myers Squibb',
    'TXN': 'Texas Instruments',
    'SBUX': 'Starbucks',
    'GS': 'Goldman Sachs',
    'USB': 'U.S. Bancorp',
    'SCHW': 'Charles Schwab',
    'AMT': 'American Tower',
    'DHR': 'Danaher',
    'UNP': 'Union Pacific',
    'CAT': 'Caterpillar',
    'UPS': 'United Parcel Service',
    'TMO': 'Thermo Fisher Scientific',
    'CME': 'CME Group',
    'TGT': 'Target',
    'DE': 'Deere & Co',
    'CVS': 'CVS Health',
    'AON': 'Aon',
    'AIG': 'American International Group',
    'BNS': 'Bank of Nova Scotia',
    'RBLX': 'Roblox',
    'SHOP': 'Shopify',
    'TWTR': 'Twitter',
    'SPOT': 'Spotify',
    'SNOW': 'Snowflake',
    'BYND': 'Beyond Meat',
    'PINS': 'Pinterest',
    'SQ': 'Square',
    'PLTR': 'Palantir Technologies'
}

# Reverse lookup for stock names
reversed_stock_dict = {v.lower(): k for k, v in stock_dict.items()}

# Function to get stock information from user input
def get_stock_info(user_input):
    user_input = user_input.lower()
    if user_input in stock_dict:
        return user_input.upper()
    elif user_input in reversed_stock_dict:
        return reversed_stock_dict[user_input]
    else:
        return None

# Streamlit app UI
st.title("Financial News Sentiment Analysis")

# Ask the user for a stock symbol or name to analyze
user_input = st.text_input(f"Enter the stock symbol or name you want to analyze ({', '.join(stock_dict.values())}): ")

if user_input:
    # Retrieve YouTube API key
    key = st.secrets["API_KEY"]  # Store your API key in Streamlit secrets
    videos = [
        'https://www.youtube.com/watch?v=RKFxWzJuQTw',
        'https://www.youtube.com/watch?v=Xa5cc8mgczc',
        'https://www.youtube.com/watch?v=EP6JqpjtUjM',
        'https://www.youtube.com/watch?v=3FnQmDld9gA'
    ]
    
    # Function to find Channel ID from a video URL
    def yt_id(url):
        yt = YouTube(url)
        return yt.channel_id

    # Function to retrieve channel data from YouTube API with date filter and pagination
    def get_channel_data(key, id):
        channel_data = []
        one_week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z'
        next_page_token = None

        while True:
            res = f'https://www.googleapis.com/youtube/v3/search?key={key}&channelId={id}&part=snippet,id&order=date&publishedAfter={one_week_ago}&maxResults=50'
            if next_page_token:
                res += f'&pageToken={next_page_token}'

            r = requests.get(res)
            data = r.json()
            if 'items' not in data:
                break

            for item in data['items']:
                publish = item['snippet']['publishedAt']
                title = item['snippet']['title']
                description = item['snippet']['description']
                channel_name = item['snippet']['channelTitle']

                video_info = {
                    'Publish Date': publish,
                    'Title': title,
                    'Description': description,
                    'Channel Name': channel_name
                }

                channel_data.append(video_info)

            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break

        df = pd.DataFrame(channel_data)
        return df

    # Loop to grab channel IDs for each video
    channel_ids = [yt_id(video) for video in videos]

    data_frames = [get_channel_data(key, id) for id in channel_ids]
    df = pd.concat(data_frames, ignore_index=True)

    # Get the corresponding stock symbol
    stock_symbol = get_stock_info(user_input)
    company_name = stock_dict.get(stock_symbol, None)
    
    if not company_name:
        st.write(f"No stock data available for the input: {user_input}")
    else:
        # Function to search for stock mentions in video titles and descriptions
        def find_stock_mentions(text, stock_symbol, company_name):
            return stock_symbol.lower() in text.lower() or company_name.lower() in text.lower()

        # Adding a column to store stock mentions
        df['Stock Mentioned'] = df.apply(lambda row: find_stock_mentions(row['Title'] + row['Description'], stock_symbol, company_name), axis=1)

        # Filter the dataframe to include only rows where the stock was mentioned
        df_filtered = df[df['Stock Mentioned']].copy()

        # Check if any stock mentions were found
        if df_filtered.empty:
            st.write(f"No mentions of {stock_symbol} ({company_name}) found in the video titles or descriptions.")
        else:
            # Perform sentiment analysis for videos mentioning the stock
            df_filtered['Sentiment Label'] = df_filtered.apply(lambda row: pipe(row['Title'] + ' ' + row['Description'])[0]['label'], axis=1)
            df_filtered['Sentiment Score'] = df_filtered.apply(lambda row: pipe(row['Title'] + ' ' + row['Description'])[0]['score'], axis=1)
            
            # Display the filtered DataFrame in a clean and formal table format
            st.write(df_filtered[['Publish Date', 'Title', 'Sentiment Label', 'Sentiment Score']].head(10))
            
            # Visualization for stock-related sentiment
            sentiment_counts = df_filtered['Sentiment Label'].value_counts()
            st.bar_chart(sentiment_counts)
