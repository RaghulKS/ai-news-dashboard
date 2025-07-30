"""
Test script for Azure Sentiment Analysis.
Run this to test the sentiment analysis functionality.
"""

import os
import sys
from azure.azure_sentiment import AzureSentimentAnalyzer
from config_secrets import AZURE_TEXT_ANALYTICS_ENDPOINT, AZURE_TEXT_ANALYTICS_KEY

def test_azure_sentiment():
    """Test the Azure Sentiment Analyzer functionality."""
    
    # Check if Azure credentials are available
    if AZURE_TEXT_ANALYTICS_ENDPOINT == "https://your-resource.cognitiveservices.azure.com/" or \
       AZURE_TEXT_ANALYTICS_KEY == "your_azure_text_analytics_key_here":
        print("❌ Azure credentials not configured in config_secrets.py")
        print("Please update your Azure credentials in config_secrets.py:")
        print("1. Go to Azure Portal > Cognitive Services > Text Analytics")
        print("2. Copy your endpoint and key")
        print("3. Update the values in secrets.py")
        return False
    
    try:
        # Initialize the analyzer
        analyzer = AzureSentimentAnalyzer()
        print("✅ Azure Sentiment Analyzer initialized successfully")
        
        # Test with sample financial headlines
        test_headlines = [
            "Apple stock surges to new record high after strong earnings",
            "Tech company faces major lawsuit over data privacy concerns", 
            "Market remains stable as investors await Fed decision",
            "Tesla shares plummet after disappointing quarterly results",
            "Bank of America reports record profits, stock jumps 5%"
        ]
        
        print(f"\n🔍 Testing sentiment analysis for {len(test_headlines)} headlines...")
        
        # Test individual analysis
        print("\n📊 Individual Analysis Results:")
        for i, headline in enumerate(test_headlines, 1):
            result = analyzer.analyze_sentiment(headline)
            sentiment = result['sentiment']
            confidence = max(result['confidence_scores'].values())
            
            # Color coding for sentiment
            if sentiment == 'positive':
                sentiment_emoji = "🟢"
            elif sentiment == 'negative':
                sentiment_emoji = "🔴"
            else:
                sentiment_emoji = "🟡"
            
            print(f"{i}. {sentiment_emoji} {headline}")
            print(f"   Sentiment: {sentiment} (confidence: {confidence:.2f})")
        
        # Test batch analysis
        print("\n📈 Batch Analysis Results:")
        batch_results = analyzer.analyze_multiple_texts(test_headlines)
        summary = analyzer.get_sentiment_summary(batch_results)
        
        print(f"Total articles analyzed: {summary['total_articles']}")
        print(f"Sentiment distribution:")
        for sentiment, count in summary['sentiment_distribution'].items():
            emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
            print(f"  {emoji} {sentiment.capitalize()}: {count}")
        print(f"Average confidence: {summary['average_confidence']:.2f}")
        print(f"Dominant sentiment: {summary['dominant_sentiment']}")
        
        # Test with news articles format
        print("\n📰 Testing with news articles format...")
        sample_articles = [
            {
                'title': 'Apple stock surges to new record high after strong earnings',
                'description': 'Apple Inc. reported better-than-expected quarterly results',
                'source': 'Financial Times',
                'published_at': '2024-01-15T10:30:00Z'
            },
            {
                'title': 'Tech company faces major lawsuit over data privacy concerns',
                'description': 'A major technology company is facing legal challenges',
                'source': 'Reuters',
                'published_at': '2024-01-15T09:15:00Z'
            }
        ]
        
        enriched_articles = analyzer.analyze_news_articles(sample_articles)
        print(f"✅ Successfully analyzed {len(enriched_articles)} articles")
        
        for article in enriched_articles:
            sentiment = article['sentiment_analysis']['sentiment']
            emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
            print(f"  {emoji} {article['title']} -> {sentiment}")
        
        print("\n🎉 All Azure sentiment analysis tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_azure_sentiment()
    sys.exit(0 if success else 1) 