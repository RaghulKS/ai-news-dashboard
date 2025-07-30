"""
Configuration settings for the AI-Powered News Sentiment Dashboard.
"""

import os
from config_secrets import (
    NEWS_API_KEY,
    AZURE_TEXT_ANALYTICS_ENDPOINT,
    AZURE_TEXT_ANALYTICS_KEY,
    AZURE_STORAGE_CONNECTION_STRING
)

# Application Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
MAX_ARTICLES_PER_QUERY = int(os.getenv('MAX_ARTICLES_PER_QUERY', '5'))
DAYS_BACK_FOR_NEWS = int(os.getenv('DAYS_BACK_FOR_NEWS', '7'))

# Model Configuration
MODEL_PATH = os.path.join('ml', 'models')
DATA_PATH = os.path.join('data')

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Validation
def validate_config():
    """Validate that required configuration is present."""
    missing_configs = []
    
    if not NEWS_API_KEY:
        missing_configs.append("NEWS_API_KEY")
    
    if not AZURE_TEXT_ANALYTICS_ENDPOINT:
        missing_configs.append("AZURE_TEXT_ANALYTICS_ENDPOINT")
    
    if not AZURE_TEXT_ANALYTICS_KEY:
        missing_configs.append("AZURE_TEXT_ANALYTICS_KEY")
    
    if missing_configs:
        print("❌ Missing required configuration:")
        for config in missing_configs:
            print(f"   - {config}")
        print("\nPlease add these to your config_secrets.py file.")
        return False
    
    return True

if __name__ == "__main__":
    print("Configuration validation:")
    if validate_config():
        print("✅ All required configuration is present")
    else:
        print("❌ Configuration validation failed") 