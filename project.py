import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load Reddit data
reddit_data = pd.read_csv('reddit_vm (1).csv')

# Load positive and negative words with scores
sentiment_dict = {}
with open('sentiment_words.txt', 'r') as file:
    for line in file:
        word, score = line.strip().split('\t')
        sentiment_dict[word] = int(score)

# Function to calculate sentiment polarity
def calculate_sentiment_polarity(text):
    # Check for NaN values
    if pd.isnull(text):
        return 'Neutral'

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    
    # Handle cases where text is not a string
    if not isinstance(text, str):
        return 'Neutral'
    
    words = [word for word in text.lower().split() if word.isalpha() and word not in stop_words]

    # Calculate sentiment score
    sentiment_score = sum(sentiment_dict.get(word, 0) for word in words)

    # Determine sentiment
    if sentiment_score == 0:
        return 'Neutral'
    elif sentiment_score > 0:
        return 'Positive'
    else:
        return 'Negative'

# Apply sentiment analysis to each comment or post
reddit_data['Sentiment'] = reddit_data['body'].apply(calculate_sentiment_polarity)

#pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Display the results
print(reddit_data[['id', 'body', 'Sentiment']])
