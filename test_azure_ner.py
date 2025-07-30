"""
Test script for Azure Named Entity Recognition (NER).
Run this to test the NER functionality.
"""

import os
import sys
from azure.azure_ner import AzureNERAnalyzer
from config_secrets import AZURE_TEXT_ANALYTICS_ENDPOINT, AZURE_TEXT_ANALYTICS_KEY

def test_azure_ner():
    """Test the Azure NER Analyzer functionality."""
    
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
        analyzer = AzureNERAnalyzer()
        print("✅ Azure NER Analyzer initialized successfully")
        
        # Test with sample financial headlines
        test_headlines = [
            "Apple stock surges to new record high after strong earnings",
            "Microsoft and Google announce partnership in cloud computing",
            "Tesla faces regulatory scrutiny from SEC over financial disclosures",
            "Bank of America reports record profits, stock jumps 5%",
            "Amazon acquires startup for $2 billion in strategic move",
            "Federal Reserve announces interest rate decision",
            "Goldman Sachs and Morgan Stanley report quarterly earnings",
            "Netflix shares fall after subscriber growth slows"
        ]
        
        print(f"\n🔍 Testing NER analysis for {len(test_headlines)} headlines...")
        
        # Test individual analysis
        print("\n📊 Individual NER Analysis Results:")
        for i, headline in enumerate(test_headlines, 1):
            result = analyzer.extract_entities(headline)
            organizations = result['organizations']
            companies = result['companies']
            total_entities = len(result['entities'])
            
            print(f"\n{i}. {headline}")
            if organizations:
                print(f"   🏢 Organizations: {', '.join(organizations)}")
            if companies:
                print(f"   👤 Companies/People: {', '.join(companies)}")
            print(f"   📈 Total entities: {total_entities}")
            
            if result.get('error'):
                print(f"   ⚠️  Error: {result['error']}")
        
        # Test batch analysis
        print("\n" + "="*60)
        print("📈 Batch NER Analysis Results:")
        batch_results = analyzer.extract_entities_batch(test_headlines)
        summary = analyzer.get_entity_summary(batch_results)
        
        print(f"Total articles analyzed: {summary['total_articles']}")
        print(f"Total entities found: {summary['total_entities']}")
        
        if summary['unique_organizations']:
            print(f"\n🏢 Unique organizations found:")
            for org in summary['unique_organizations']:
                print(f"   • {org}")
        
        if summary['unique_companies']:
            print(f"\n👤 Unique companies/people found:")
            for company in summary['unique_companies']:
                print(f"   • {company}")
        
        print(f"\n📊 Entity categories:")
        for category, count in summary['entity_categories'].items():
            print(f"   • {category}: {count}")
        
        print(f"\n🔥 Most mentioned entities:")
        for item in summary['most_mentioned_entities'][:5]:
            print(f"   • {item['entity']}: {item['count']} mentions")
        
        # Test with news articles format
        print("\n" + "="*60)
        print("📰 Testing with news articles format...")
        sample_articles = [
            {
                'title': 'Apple stock surges to new record high after strong earnings',
                'description': 'Apple Inc. reported better-than-expected quarterly results',
                'source': 'Financial Times',
                'published_at': '2024-01-15T10:30:00Z'
            },
            {
                'title': 'Microsoft and Google announce partnership in cloud computing',
                'description': 'Tech giants collaborate on new cloud infrastructure',
                'source': 'Reuters',
                'published_at': '2024-01-15T09:15:00Z'
            },
            {
                'title': 'Tesla faces regulatory scrutiny from SEC over financial disclosures',
                'description': 'Electric vehicle maker under investigation',
                'source': 'Bloomberg',
                'published_at': '2024-01-15T08:45:00Z'
            }
        ]
        
        enriched_articles = analyzer.analyze_news_articles(sample_articles)
        print(f"✅ Successfully analyzed {len(enriched_articles)} articles")
        
        for article in enriched_articles:
            ner_analysis = article['ner_analysis']
            organizations = ner_analysis['organizations']
            companies = ner_analysis['companies']
            
            print(f"\n📰 {article['title']}")
            if organizations:
                print(f"   🏢 Organizations: {', '.join(organizations)}")
            if companies:
                print(f"   👤 Companies: {', '.join(companies)}")
        
        # Test financial entity filtering
        print("\n" + "="*60)
        print("💰 Testing financial entity filtering...")
        
        # Get all entities from batch results
        all_entities = []
        for result in batch_results:
            all_entities.extend(result['entities'])
        
        financial_entities = analyzer.filter_financial_entities(all_entities)
        print(f"Found {len(financial_entities)} financial-related entities:")
        
        for entity in financial_entities[:10]:  # Show first 10
            print(f"   • {entity['text']} ({entity['category']})")
        
        print("\n🎉 All Azure NER tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_azure_ner()
    sys.exit(0 if success else 1) 