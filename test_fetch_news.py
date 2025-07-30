"""
Test script for the news fetching module.
Run this to test the NewsFetcher functionality.
"""

import os
import sys
from utils.fetch_news import NewsFetcher
from config_secrets import NEWS_API_KEY

def test_news_fetcher():
    """Test the NewsFetcher class functionality."""
    
    # Check if API key is available
    if not NEWS_API_KEY:
        print("❌ NEWS_API_KEY not found in config_secrets.py.")
        print("Please add your NewsAPI key to config_secrets.py:")
        print("NEWS_API_KEY = 'your_api_key_here'")
        return False
    
    try:
        # Initialize the fetcher
        fetcher = NewsFetcher()
        print("✅ NewsFetcher initialized successfully")
        
        # Test fetching news for a popular stock
        print("\n🔍 Testing news fetch for 'AAPL'...")
        articles = fetcher.fetch_news("AAPL", count=3)
        
        if articles:
            print(f"✅ Successfully retrieved {len(articles)} articles")
            print("\n📰 Sample articles:")
            for i, article in enumerate(articles[:2], 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Source: {article['source']}")
                print(f"   Published: {article['published_at']}")
                print(f"   URL: {article['url'][:80]}...")
        else:
            print("❌ No articles retrieved")
            return False
        
        # Test saving to file
        print("\n💾 Testing file save functionality...")
        filepath = fetcher.save_articles_to_file(articles)
        if filepath:
            print(f"✅ Articles saved to: {filepath}")
        else:
            print("❌ Failed to save articles")
            return False
        
        print("\n🎉 All tests passed! News fetching module is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_news_fetcher()
    sys.exit(0 if success else 1) 