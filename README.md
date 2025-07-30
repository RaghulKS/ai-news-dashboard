A real-time Streamlit dashboard that analyzes financial news sentiment and predicts potential market movements using Azure AI services and custom ML models. Get instant insights into how news affects stock prices with beautiful visualizations and AI-powered predictions.


This dashboard transforms financial news into actionable market insights by:

- Fetching real-time financial news** from NewsAPI based on your queries
- Analyzing sentiment** using Azure Cognitive Services (positive/neutral/negative)
- Extracting companies and organizations** mentioned in the news
- Predicting stock movements** using a custom ML model (Up/Down/Flat)
- Presenting everything** in a beautiful, interactive Streamlit interface

Key Features

Smart News Analysis
- Real-time news fetching from NewsAPI
- Intelligent filtering for financial relevance
- Support for any stock ticker or company name
- Configurable number of articles (1-10)

Advanced Sentiment Analysis
- Azure Cognitive Services integration
- Three-way sentiment classification (Positive/Neutral/Negative)
- Confidence scores for each sentiment
- Beautiful dual visualization (pie chart + bar chart)

Named Entity Recognition
- Automatic extraction of companies and organizations
- Financial entity filtering
- Entity confidence scoring
- Clean presentation in analysis cards

ML-Powered Predictions
- Custom Random Forest classifier
- Stock movement prediction (Up/Down/Flat)
- Confidence scores for predictions
- Feature importance analysis

Beautiful Visualizations**
- Interactive charts with Plotly
- Sentiment distribution charts
- Stock movement prediction charts
- Real-time metrics dashboard
- Export results to CSV

Enhanced User Experience**
- Modern gradient design
- Responsive layout
- Bigger fonts and better colors
- Visual separators and cards
- Comprehensive error handling

Tech Stack

Frontend & UI
- Streamlit - Interactive web dashboard
- Plotly - Interactive charts and visualizations
- Custom CSS - Modern styling and gradients

Backend & APIs
- Python 3.8+ - Core programming language
- NewsAPI - Financial news data
- Azure Cognitive Services - Sentiment analysis & NER
- Azure Storage - Data persistence (optional)

Machine Learning
- scikit-learn - ML algorithms and preprocessing
- XGBoost - Gradient boosting for predictions
- TF-IDF - Text feature extraction
- joblib - Model serialization

Data Processing
- pandas - Data manipulation and analysis
- numpy - Numerical computations
- requests - HTTP API calls

Development & Testing
- pytest - Unit and integration testing
- python-dotenv - Environment configuration
- black - Code formatting
- flake8 - Code linting


 Quick Start

Prerequisites
- Python 3.8 or higher
- NewsAPI account (free tier available)
- Azure Cognitive Services account


1. Set Up API Keys
   
Update the `config_secrets.py` file in the project root:

NewsAPI Configuration
NEWS_API_KEY = "your_news_api_key_here"

Azure Cognitive Services
AZURE_TEXT_ANALYTICS_ENDPOINT = "https://your-resource.cognitiveservices.azure.com/"
AZURE_TEXT_ANALYTICS_KEY = "your_azure_text_analytics_key_here"

Optional: Azure Storage
AZURE_STORAGE_CONNECTION_STRING = "your_azure_storage_connection_string_here"


2. Train the ML Model
   
bash
python ml/train_model.py


4. Run the Dashboard
   
bash
streamlit run app.py


6. Open Your Browser
   
Navigate to `http://localhost:****` to see the dashboard!

API Setup Guide

NewsAPI Setup
1. Visit [NewsAPI.org](https://newsapi.org/)
2. Sign up for a free account
3. Copy your API key
4. Add it to your `config_secrets.py` file

Azure Cognitive Services Setup
1. Go to [Azure Portal](https://portal.azure.com/)
2. Create a new Cognitive Services resource
3. Enable Text Analytics service
4. Copy the endpoint and key
5. Add them to your `config_secrets.py` file



 Project Structure

ai-news-dashboard/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── SETUP.md                 # Detailed setup guide
├── README.md                # This file
├── test_*.py                # Test scripts
├── utils/
│   └── fetch_news.py        # News fetching module
├── azure/
│   ├── azure_sentiment.py   # Azure sentiment analysis
│   └── azure_ner.py         # Azure NER
├── ml/
│   ├── train_model.py       # ML model training
│   ├── model.py            # ML model interface
│   └── models/             # Saved model files
└── data/                   # Data storage directory


 Configuration

Environment Variables
- `NEWS_API_KEY`: Your NewsAPI key
- `AZURE_TEXT_ANALYTICS_ENDPOINT`: Azure endpoint
- `AZURE_TEXT_ANALYTICS_KEY`: Azure key
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `MAX_ARTICLES_PER_QUERY`: Maximum articles to fetch (default: 5)

Customization
- Modify `ml/train_model.py` to adjust ML model parameters
- Update `utils/fetch_news.py` to change news filtering logic
- Customize charts in `app.py` for different visualizations

