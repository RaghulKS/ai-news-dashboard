import os
import logging
import requests
from typing import Dict, List, Optional, Tuple
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
import json
from config_secrets import AZURE_TEXT_ANALYTICS_ENDPOINT, AZURE_TEXT_ANALYTICS_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureSentimentAnalyzer:
    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        self.endpoint = endpoint or AZURE_TEXT_ANALYTICS_ENDPOINT
        self.key = key or AZURE_TEXT_ANALYTICS_KEY
        
        if not self.endpoint or not self.key:
            raise ValueError(
                "Azure Text Analytics credentials are required. "
                "Please add AZURE_TEXT_ANALYTICS_ENDPOINT and AZURE_TEXT_ANALYTICS_KEY "
                "to your config_secrets.py file."
            )
        
        try:
            self.client = TextAnalyticsClient(
                endpoint=self.endpoint, 
                credential=AzureKeyCredential(self.key)
            )
            logger.info("Azure Text Analytics client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")
            raise

    def analyze_sentiment(self, text: str) -> Dict:
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'sentiment': 'neutral',
                    'confidence_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                    'sentiment_score': 0.5
                }
            
            result = self.client.analyze_sentiment([text])[0]
            
            sentiment = result.sentiment
            confidence_scores = result.confidence_scores
            
            return {
                'sentiment': sentiment,
                'confidence_scores': {
                    'positive': confidence_scores.positive,
                    'negative': confidence_scores.negative,
                    'neutral': confidence_scores.neutral
                },
                'sentiment_score': confidence_scores.positive
            }
            
        except AzureError as e:
            logger.error(f"Azure API error: {e}")
            return {
                'sentiment': 'neutral',
                'confidence_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'sentiment_score': 0.5
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'sentiment_score': 0.5
            }

    def analyze_news_articles(self, articles: List[Dict]) -> List[Dict]:
        try:
            if not articles:
                logger.warning("No articles provided for sentiment analysis")
                return []
            
            headlines = [article.get('title', '') for article in articles]
            
            results = []
            for i, headline in enumerate(headlines):
                sentiment_result = self.analyze_sentiment(headline)
                
                article_with_sentiment = articles[i].copy()
                article_with_sentiment['sentiment_analysis'] = sentiment_result
                
                results.append(article_with_sentiment)
                
                logger.info(f"Analyzed sentiment for article {i+1}: {sentiment_result['sentiment']}")
            
            logger.info(f"Completed sentiment analysis for {len(results)} articles")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing news articles: {e}")
            return []

    def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        try:
            if not texts:
                return []
            
            results = []
            for text in texts:
                result = self.analyze_sentiment(text)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return []


def main():
    try:
        analyzer = AzureSentimentAnalyzer()
        
        test_texts = [
            "Apple stock surges to new record high after strong earnings",
            "Tesla faces regulatory scrutiny, stock drops",
            "Microsoft announces new product line"
        ]
        
        print("Testing sentiment analysis...")
        for text in test_texts:
            result = analyzer.analyze_sentiment(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence_scores']}")
            print()
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 