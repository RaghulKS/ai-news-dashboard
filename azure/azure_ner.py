import os
import logging
import requests
from typing import Dict, List, Optional, Set
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
import json
from config_secrets import AZURE_TEXT_ANALYTICS_ENDPOINT, AZURE_TEXT_ANALYTICS_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureNERAnalyzer:
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
            logger.info("Azure NER client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")
            raise

    def extract_entities(self, text: str) -> Dict:
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'organizations': [],
                    'companies': [],
                    'people': [],
                    'locations': []
                }
            
            result = self.client.recognize_entities([text])[0]
            
            organizations = []
            companies = []
            people = []
            locations = []
            
            for entity in result.entities:
                if entity.category == "Organization":
                    organizations.append(entity.text)
                elif entity.category == "Person":
                    people.append(entity.text)
                elif entity.category == "Location":
                    locations.append(entity.text)
            
            companies = list(set(organizations))
            
            return {
                'organizations': list(set(organizations)),
                'companies': companies,
                'people': list(set(people)),
                'locations': list(set(locations))
            }
            
        except AzureError as e:
            logger.error(f"Azure API error: {e}")
            return {
                'organizations': [],
                'companies': [],
                'people': [],
                'locations': []
            }
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {
                'organizations': [],
                'companies': [],
                'people': [],
                'locations': []
            }

    def analyze_news_articles(self, articles: List[Dict]) -> List[Dict]:
        try:
            if not articles:
                logger.warning("No articles provided for NER analysis")
                return []
            
            headlines = [article.get('title', '') for article in articles]
            
            results = []
            for i, headline in enumerate(headlines):
                ner_result = self.extract_entities(headline)
                
                article_with_ner = articles[i].copy()
                article_with_ner['ner_analysis'] = ner_result
                
                results.append(article_with_ner)
                
                logger.info(f"Extracted entities for article {i+1}: {len(ner_result['organizations'])} organizations")
            
            logger.info(f"Completed NER analysis for {len(results)} articles")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing news articles: {e}")
            return []

    def batch_extract_entities(self, texts: List[str]) -> List[Dict]:
        try:
            if not texts:
                return []
            
            results = []
            for text in texts:
                result = self.extract_entities(text)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch entity extraction: {e}")
            return []


def main():
    try:
        analyzer = AzureNERAnalyzer()
        
        test_texts = [
            "Apple Inc. CEO Tim Cook announced new products at the company headquarters in Cupertino",
            "Microsoft Corporation and Google LLC are competing in the cloud computing market",
            "Tesla CEO Elon Musk tweeted about the company's latest earnings report"
        ]
        
        print("Testing NER analysis...")
        for text in test_texts:
            result = analyzer.extract_entities(text)
            print(f"Text: {text}")
            print(f"Organizations: {result['organizations']}")
            print(f"Companies: {result['companies']}")
            print(f"People: {result['people']}")
            print()
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 