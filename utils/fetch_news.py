import os
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from config_secrets import NEWS_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NEWS_API_KEY
        if not self.api_key:
            raise ValueError("NewsAPI key is required. Please add NEWS_API_KEY to your config_secrets.py file.")
        
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query: str, count: int = 5, days_back: int = 7) -> List[Dict]:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            financial_keywords = [
                "earnings", "revenue", "profit", "loss", "quarterly", "annual",
                "stock", "shares", "trading", "market", "investment", "finance",
                "analyst", "upgrade", "downgrade", "target", "price", "valuation",
                "merger", "acquisition", "IPO", "dividend", "buyback", "guidance",
                "forecast", "outlook", "performance", "growth", "decline", "rally",
                "crash", "surge", "plunge", "jump", "drop", "rise", "fall"
            ]
            
            search_query = f'"{query}" AND ('
            search_query += ' OR '.join(f'"{keyword}"' for keyword in financial_keywords[:10])
            search_query += ')'
            
            params = {
                'q': search_query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(count * 2, 20),
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'apiKey': self.api_key
            }
            
            logger.info(f"Fetching news for query: {query}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            logger.info(f"Retrieved {len(articles)} articles for query: {query}")
            
            processed_articles = []
            for article in articles:
                if len(article.get('title', '')) < 20:
                    continue
                
                if not article.get('source', {}).get('name'):
                    continue
                
                if not article.get('url'):
                    continue
                
                processed_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'query': query
                }
                processed_articles.append(processed_article)
                
                if len(processed_articles) >= count:
                    break
            
            logger.info(f"Processed {len(processed_articles)} quality articles for query: {query}")
            return processed_articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching news: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while fetching news: {e}")
            return []

    def fetch_top_headlines(self, category: str = "business", country: str = "us", count: int = 5) -> List[Dict]:
        try:
            params = {
                'category': category,
                'country': country,
                'pageSize': count,
                'apiKey': self.api_key
            }
            
            response = requests.get("https://newsapi.org/v2/top-headlines", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            processed_articles = []
            
            for article in articles:
                if len(article.get('title', '')) < 20:
                    continue
                
                processed_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'query': f"{category} headlines"
                }
                processed_articles.append(processed_article)
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error fetching top headlines: {e}")
            return []


def main():
    try:
        fetcher = NewsFetcher()
        
        print("Testing news fetching...")
        articles = fetcher.fetch_news("AAPL", count=3)
        
        if articles:
            print(f"Found {len(articles)} articles:")
            for i, article in enumerate(articles, 1):
                print(f"{i}. {article['title']}")
                print(f"   Source: {article['source']}")
                print(f"   URL: {article['url']}")
                print()
        else:
            print("No articles found.")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please add NEWS_API_KEY to your config_secrets.py file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 