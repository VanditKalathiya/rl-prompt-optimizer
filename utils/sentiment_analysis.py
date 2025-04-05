from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    """
    Returns sentiment score in range [-1, 1]
    """
    scores = analyzer.polarity_scores(text)
    return scores['compound']  # compound is the overall score
