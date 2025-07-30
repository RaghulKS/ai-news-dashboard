"""
Test script for Streamlit App Components.
Run this to test the app functionality without launching the UI.
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.fetch_news import NewsFetcher
from azure.azure_sentiment import AzureSentimentAnalyzer
from azure.azure_ner import AzureNERAnalyzer
from ml.model import StockPredictor
from config_secrets import NEWS_API_KEY, AZURE_TEXT_ANALYTICS_ENDPOINT, AZURE_TEXT_ANALYTICS_KEY

def test_app_components():
    """Test all components used in the Streamlit app."""
    
    print("🚀 Testing Streamlit App Components")
    print("=" * 50)
    
    # Check if API keys are available
    if (NEWS_API_KEY == "your_news_api_key_here" or 
        AZURE_TEXT_ANALYTICS_ENDPOINT == "https://your-resource.cognitiveservices.azure.com/" or
        AZURE_TEXT_ANALYTICS_KEY == "your_azure_text_analytics_key_here"):
        print("❌ API credentials not properly configured in secrets.py")
        print("Please update your credentials in secrets.py")
        return False
    
    try:
        # Test component initialization
        print("\n🔧 Testing component initialization...")
        
        news_fetcher = NewsFetcher()
        print("✅ NewsFetcher initialized")
        
        sentiment_analyzer = AzureSentimentAnalyzer()
        print("✅ AzureSentimentAnalyzer initialized")
        
        ner_analyzer = AzureNERAnalyzer()
        print("✅ AzureNERAnalyzer initialized")
        
        stock_predictor = StockPredictor()
        print("✅ StockPredictor initialized")
        
        # Test complete pipeline
        print("\n📊 Testing complete analysis pipeline...")
        
        # Test query
        query = "AAPL"
        count = 3  # Smaller count for testing
        
        # Step 1: Fetch news
        print("📰 Fetching news...")
        articles = news_fetcher.fetch_news(query, count=count)
        
        if not articles:
            print("❌ No articles fetched. Check your NewsAPI key.")
            return False
        
        print(f"✅ Fetched {len(articles)} articles")
        
        # Step 2: Analyze sentiment
        print("🎭 Analyzing sentiment...")
        articles_with_sentiment = sentiment_analyzer.analyze_news_articles(articles)
        print(f"✅ Analyzed sentiment for {len(articles_with_sentiment)} articles")
        
        # Step 3: Extract entities
        print("🏢 Extracting entities...")
        articles_with_ner = ner_analyzer.analyze_news_articles(articles_with_sentiment)
        print(f"✅ Extracted entities for {len(articles_with_ner)} articles")
        
        # Step 4: Predict stock movement
        print("🤖 Predicting stock movement...")
        predictions = stock_predictor.predict_from_articles(articles_with_ner)
        print(f"✅ Made predictions for {len(predictions)} articles")
        
        # Combine results (same as in app.py)
        results = []
        for i, (article, prediction) in enumerate(zip(articles_with_ner, predictions)):
            result = {
                'index': i + 1,
                'headline': article['title'],
                'source': article['source'],
                'published_at': article['published_at'],
                'url': article['url'],
                'sentiment': article['sentiment_analysis']['sentiment'],
                'sentiment_confidence': max(article['sentiment_analysis']['confidence_scores'].values()),
                'sentiment_scores': article['sentiment_analysis']['confidence_scores'],
                'entities': {
                    'organizations': article['ner_analysis']['organizations'],
                    'companies': article['ner_analysis']['companies']
                },
                'predicted_movement': prediction['predicted_movement'],
                'prediction_confidence': prediction['max_confidence'],
                'prediction_scores': prediction['confidence_scores']
            }
            results.append(result)
        
        # Display results
        print("\n📊 Analysis Results:")
        print("-" * 50)
        
        for result in results:
            print(f"\n{result['index']}. {result['headline']}")
            print(f"   Source: {result['source']}")
            print(f"   Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.2f})")
            print(f"   Prediction: {result['predicted_movement']} (confidence: {result['prediction_confidence']:.2f})")
            
            if result['entities']['organizations']:
                print(f"   Organizations: {', '.join(result['entities']['organizations'])}")
            if result['entities']['companies']:
                print(f"   Companies: {', '.join(result['entities']['companies'])}")
        
        # Test metrics calculation
        print("\n📈 Testing metrics calculation...")
        
        # Calculate metrics (same as in app.py)
        total_articles = len(results)
        sentiment_counts = {}
        movement_counts = {}
        total_confidence = 0
        
        for result in results:
            # Sentiment distribution
            sentiment = result['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # Movement distribution
            movement = result['predicted_movement']
            movement_counts[movement] = movement_counts.get(movement, 0) + 1
            
            # Average confidence
            total_confidence += result['prediction_confidence']
        
        avg_confidence = total_confidence / total_articles if total_articles > 0 else 0
        
        print(f"   Total articles: {total_articles}")
        print(f"   Sentiment distribution: {sentiment_counts}")
        print(f"   Movement distribution: {movement_counts}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        
        # Test data export format
        print("\n💾 Testing data export format...")
        
        export_data = []
        for result in results:
            export_data.append({
                'Headline': result['headline'],
                'Source': result['source'],
                'Published': result['published_at'],
                'Sentiment': result['sentiment'],
                'Sentiment_Confidence': result['sentiment_confidence'],
                'Predicted_Movement': result['predicted_movement'],
                'Prediction_Confidence': result['prediction_confidence'],
                'Organizations': ', '.join(result['entities']['organizations']),
                'Companies': ', '.join(result['entities']['companies'])
            })
        
        print(f"   Export data format: {len(export_data)} rows")
        print(f"   Export columns: {list(export_data[0].keys()) if export_data else 'No data'}")
        
        print("\n🎉 All Streamlit app component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_imports():
    """Test that all app imports work correctly."""
    
    print("\n📦 Testing App Imports")
    print("=" * 30)
    
    try:
        # Test all imports used in app.py
        import streamlit as st
        print("✅ streamlit imported")
        
        import pandas as pd
        print("✅ pandas imported")
        
        import plotly.express as px
        print("✅ plotly.express imported")
        
        import plotly.graph_objects as go
        print("✅ plotly.graph_objects imported")
        
        from datetime import datetime
        print("✅ datetime imported")
        
        # Test our custom modules
        from utils.fetch_news import NewsFetcher
        print("✅ NewsFetcher imported")
        
        from azure.azure_sentiment import AzureSentimentAnalyzer
        print("✅ AzureSentimentAnalyzer imported")
        
        from azure.azure_ner import AzureNERAnalyzer
        print("✅ AzureNERAnalyzer imported")
        
        from ml.model import StockPredictor
        print("✅ StockPredictor imported")
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    # Test imports first
    success1 = test_app_imports()
    
    # Test components
    success2 = test_app_components()
    
    if success1 and success2:
        print("\n🎉 All Streamlit app tests passed!")
        print("\n🚀 You can now run the app with:")
        print("   streamlit run app.py")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 