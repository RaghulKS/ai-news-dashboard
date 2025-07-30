"""
Integration Test Script

This script tests the complete pipeline: news fetching + sentiment analysis + NER.
Run this to verify all components work together.
"""

import os
import sys
import json
from datetime import datetime
from utils.fetch_news import NewsFetcher
from azure.azure_sentiment import AzureSentimentAnalyzer
from azure.azure_ner import AzureNERAnalyzer
from config_secrets import NEWS_API_KEY, AZURE_TEXT_ANALYTICS_ENDPOINT, AZURE_TEXT_ANALYTICS_KEY

def test_integration():
    """Test the complete news analysis pipeline."""
    
    print("🚀 Testing Complete News Analysis Pipeline")
    print("=" * 60)
    
    # Check if credentials are available
    if (NEWS_API_KEY == "your_news_api_key_here" or 
        AZURE_TEXT_ANALYTICS_ENDPOINT == "https://your-resource.cognitiveservices.azure.com/" or
        AZURE_TEXT_ANALYTICS_KEY == "your_azure_text_analytics_key_here"):
        print("❌ API credentials not properly configured in config_secrets.py")
        print("Please update your credentials in config_secrets.py:")
        print("1. NEWS_API_KEY - Get from https://newsapi.org/")
        print("2. AZURE_TEXT_ANALYTICS_ENDPOINT - From Azure Portal")
        print("3. AZURE_TEXT_ANALYTICS_KEY - From Azure Portal")
        return False
    
    try:
        # Initialize all components
        print("\n🔧 Initializing components...")
        
        news_fetcher = NewsFetcher()
        print("✅ NewsFetcher initialized")
        
        sentiment_analyzer = AzureSentimentAnalyzer()
        print("✅ AzureSentimentAnalyzer initialized")
        
        ner_analyzer = AzureNERAnalyzer()
        print("✅ AzureNERAnalyzer initialized")
        
        # Test with a popular stock
        test_query = "AAPL"
        print(f"\n📰 Fetching news for '{test_query}'...")
        
        # Step 1: Fetch news
        articles = news_fetcher.fetch_news(test_query, count=3)
        
        if not articles:
            print("❌ No articles fetched. Check your NewsAPI key and internet connection.")
            return False
        
        print(f"✅ Fetched {len(articles)} articles")
        
        # Display fetched articles
        print("\n📰 Fetched Articles:")
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Published: {article['published_at']}")
            print(f"   URL: {article['url'][:80]}...")
        
        # Step 2: Analyze sentiment
        print(f"\n🎭 Analyzing sentiment for {len(articles)} articles...")
        enriched_articles = sentiment_analyzer.analyze_news_articles(articles)
        
        # Step 3: Extract entities
        print(f"\n🏢 Extracting entities for {len(articles)} articles...")
        final_articles = ner_analyzer.analyze_news_articles(enriched_articles)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 ANALYSIS RESULTS")
        print("=" * 60)
        
        for i, article in enumerate(final_articles, 1):
            print(f"\n📰 Article {i}: {article['title']}")
            print(f"   Source: {article['source']}")
            
            # Sentiment results
            sentiment_analysis = article.get('sentiment_analysis', {})
            if sentiment_analysis:
                sentiment = sentiment_analysis.get('sentiment', 'unknown')
                confidence = max(sentiment_analysis.get('confidence_scores', {}).values(), default=0)
                
                # Color coding
                if sentiment == 'positive':
                    sentiment_emoji = "🟢"
                elif sentiment == 'negative':
                    sentiment_emoji = "🔴"
                else:
                    sentiment_emoji = "🟡"
                
                print(f"   {sentiment_emoji} Sentiment: {sentiment} (confidence: {confidence:.2f})")
            
            # NER results
            ner_analysis = article.get('ner_analysis', {})
            if ner_analysis:
                organizations = ner_analysis.get('organizations', [])
                companies = ner_analysis.get('companies', [])
                
                if organizations:
                    print(f"   🏢 Organizations: {', '.join(organizations)}")
                if companies:
                    print(f"   👤 Companies/People: {', '.join(companies)}")
        
        # Generate summary statistics
        print("\n" + "=" * 60)
        print("📈 SUMMARY STATISTICS")
        print("=" * 60)
        
        # Sentiment summary
        sentiment_results = [article.get('sentiment_analysis', {}) for article in final_articles]
        sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentiment_results)
        
        print(f"\n🎭 Sentiment Analysis:")
        print(f"   Total articles: {sentiment_summary['total_articles']}")
        print(f"   Average confidence: {sentiment_summary['average_confidence']:.2f}")
        print(f"   Dominant sentiment: {sentiment_summary['dominant_sentiment']}")
        
        sentiment_dist = sentiment_summary['sentiment_distribution']
        for sentiment, count in sentiment_dist.items():
            emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
            print(f"   {emoji} {sentiment.capitalize()}: {count}")
        
        # NER summary
        ner_results = [article.get('ner_analysis', {}) for article in final_articles]
        ner_summary = ner_analyzer.get_entity_summary(ner_results)
        
        print(f"\n🏢 Entity Recognition:")
        print(f"   Total entities found: {ner_summary['total_entities']}")
        print(f"   Unique organizations: {len(ner_summary['unique_organizations'])}")
        print(f"   Unique companies: {len(ner_summary['unique_companies'])}")
        
        if ner_summary['unique_organizations']:
            print(f"   Organizations: {', '.join(ner_summary['unique_organizations'])}")
        if ner_summary['unique_companies']:
            print(f"   Companies: {', '.join(ner_summary['unique_companies'])}")
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integration_test_results_{timestamp}.json"
        
        # Prepare data for saving
        results_data = {
            'test_timestamp': datetime.now().isoformat(),
            'query': test_query,
            'articles': final_articles,
            'sentiment_summary': sentiment_summary,
            'ner_summary': ner_summary
        }
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filepath}")
        
        print("\n🎉 Integration test completed successfully!")
        print("✅ News fetching: Working")
        print("✅ Sentiment analysis: Working")
        print("✅ Entity recognition: Working")
        print("✅ Data persistence: Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1) 