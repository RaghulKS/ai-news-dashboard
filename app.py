import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.fetch_news import NewsFetcher
from azure.azure_sentiment import AzureSentimentAnalyzer
from azure.azure_ner import AzureNERAnalyzer
from ml.model import StockPredictor
from config_secrets import NEWS_API_KEY, AZURE_TEXT_ANALYTICS_ENDPOINT, AZURE_TEXT_ANALYTICS_KEY

st.set_page_config(
    page_title="AI News Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .prediction-up {
        color: #27ae60;
        font-weight: bold;
        font-size: 1.2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .prediction-down {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .prediction-flat {
        color: #95a5a6;
        font-weight: bold;
        font-size: 1.2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sentiment-positive {
        color: #27ae60;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .sentiment-neutral {
        color: #95a5a6;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
    }
    
    .article-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #e9ecef;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .separator {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = None

def check_api_keys():
    missing_keys = []
    
    if not NEWS_API_KEY:
        missing_keys.append("NEWS_API_KEY")
    
    if not AZURE_TEXT_ANALYTICS_ENDPOINT:
        missing_keys.append("AZURE_TEXT_ANALYTICS_ENDPOINT")
    
    if not AZURE_TEXT_ANALYTICS_KEY:
        missing_keys.append("AZURE_TEXT_ANALYTICS_KEY")
    
    return missing_keys

def initialize_components():
    try:
        missing_keys = check_api_keys()
        if missing_keys:
            st.error(f"Missing API keys: {', '.join(missing_keys)}")
            st.info("Please add the required API keys to your config_secrets.py file.")
            return None, None, None, None
        
        news_fetcher = NewsFetcher()
        sentiment_analyzer = AzureSentimentAnalyzer()
        ner_analyzer = AzureNERAnalyzer()
        stock_predictor = StockPredictor()
        
        return news_fetcher, sentiment_analyzer, ner_analyzer, stock_predictor
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None, None

def analyze_news(query: str, count: int = 5):
    try:
        news_fetcher, sentiment_analyzer, ner_analyzer, stock_predictor = initialize_components()
        
        if not all([news_fetcher, sentiment_analyzer, ner_analyzer, stock_predictor]):
            return None
        
        with st.spinner("📰 Fetching latest news..."):
            articles = news_fetcher.fetch_news(query, count=count)
            if not articles:
                st.error(f"No news found for '{query}'. Try a different query or check your NewsAPI key.")
                return None
            
            if len(articles) == 0:
                st.error(f"No articles returned for '{query}'. The API might be rate-limited or the query returned no results.")
                return None
        
        with st.spinner("🎭 Analyzing sentiment..."):
            articles_with_sentiment = sentiment_analyzer.analyze_news_articles(articles)
            if not articles_with_sentiment:
                st.error("Sentiment analysis failed. Check your Azure credentials.")
                return None
        
        with st.spinner("🏢 Extracting entities..."):
            articles_with_ner = ner_analyzer.analyze_news_articles(articles_with_sentiment)
            if not articles_with_ner:
                st.error("Entity extraction failed. Check your Azure credentials.")
                return None
        
        with st.spinner("🤖 Predicting stock movement..."):
            predictions = stock_predictor.predict_from_articles(articles_with_ner)
            if not predictions:
                st.warning("Using sentiment-based predictions (ML model not available)")
                predictions = []
                for article in articles_with_ner:
                    sentiment = article['sentiment_analysis']['sentiment']
                    confidence = max(article['sentiment_analysis']['confidence_scores'].values())
                    
                    if sentiment == 'positive':
                        movement = 'Up'
                    elif sentiment == 'negative':
                        movement = 'Down'
                    else:
                        movement = 'Flat'
                    
                    predictions.append({
                        'predicted_movement': movement,
                        'max_confidence': confidence,
                        'confidence_scores': {'Up': 0.33, 'Down': 0.33, 'Flat': 0.34}
                    })
        
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
        
        return results
        
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return None

def display_metrics(results):
    if not results:
        return
    
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">📊 Analysis Summary</h2>', unsafe_allow_html=True)
    
    total_articles = len(results)
    sentiment_counts = {}
    movement_counts = {}
    total_sentiment_confidence = 0
    total_prediction_confidence = 0
    total_entities = 0
    
    for result in results:
        sentiment = result['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        total_sentiment_confidence += result['sentiment_confidence']
        
        movement = result['predicted_movement']
        movement_counts[movement] = movement_counts.get(movement, 0) + 1
        total_prediction_confidence += result['prediction_confidence']
        
        entities = result['entities']
        total_entities += len(entities.get('organizations', [])) + len(entities.get('companies', []))
    
    avg_sentiment_confidence = total_sentiment_confidence / total_articles
    avg_prediction_confidence = total_prediction_confidence / total_articles
    avg_entities_per_article = total_entities / total_articles
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📰 Articles Analyzed", total_articles)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "N/A"
        st.metric("🎭 Dominant Sentiment", dominant_sentiment.title())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        dominant_movement = max(movement_counts, key=movement_counts.get) if movement_counts else "N/A"
        st.metric("📈 Dominant Prediction", dominant_movement)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Avg Confidence", f"{avg_prediction_confidence:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Sentiment Confidence", f"{avg_sentiment_confidence:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏢 Avg Entities/Article", f"{avg_entities_per_article:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔍 Total Entities Found", total_entities)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Sentiment Distribution:**")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_articles) * 100
            emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
            st.markdown(f"{emoji} {sentiment.title()}: {count} ({percentage:.1f}%)")
    
    with col2:
        st.markdown("**📈 Prediction Distribution:**")
        for movement, count in movement_counts.items():
            percentage = (count / total_articles) * 100
            emoji = "📈" if movement == "Up" else "📉" if movement == "Down" else "➡️"
            st.markdown(f"{emoji} {movement}: {count} ({percentage:.1f}%)")

def display_sentiment_charts(results):
    if not results:
        return
    
    sentiment_counts = {}
    for result in results:
        sentiment = result['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    if sentiment_counts:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_pie = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title="Sentiment Distribution (Pie Chart)",
                color_discrete_map={
                    'positive': '#27ae60',
                    'negative': '#e74c3c',
                    'neutral': '#95a5a6'
                }
            )
            fig_pie.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_bar = px.bar(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values()),
                title="Sentiment Distribution (Bar Chart)",
                color=list(sentiment_counts.keys()),
                color_discrete_map={
                    'positive': '#27ae60',
                    'negative': '#e74c3c',
                    'neutral': '#95a5a6'
                }
            )
            fig_bar.update_layout(height=400, showlegend=False)
            fig_bar.update_xaxes(title_text="Sentiment")
            fig_bar.update_yaxes(title_text="Count")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def display_prediction_chart(results):
    if not results:
        return
    
    movement_counts = {}
    for result in results:
        movement = result['predicted_movement']
        movement_counts[movement] = movement_counts.get(movement, 0) + 1
    
    if movement_counts:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = px.bar(
            x=list(movement_counts.keys()),
            y=list(movement_counts.values()),
            title="Stock Movement Predictions",
            color=list(movement_counts.keys()),
            color_discrete_map={
                'Up': '#27ae60',
                'Down': '#e74c3c',
                'Flat': '#95a5a6'
            }
        )
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Predicted Movement")
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_article_card(result):
    st.markdown('<div class="article-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{result['index']}. {result['headline']}**")
    with col2:
        st.markdown(f"*Source: {result['source']}*")
    
    st.markdown(f"📅 **Published:** {result['published_at']}")
    if result['url']:
        st.markdown(f"🔗 [Read Full Article]({result['url']})")
    
    st.markdown("---")
    st.markdown("**🎭 Sentiment Analysis:**")
    
    sentiment = result['sentiment']
    confidence = result['sentiment_confidence']
    
    if sentiment == 'positive':
        sentiment_color = "🟢"
        sentiment_class = "sentiment-positive"
    elif sentiment == 'negative':
        sentiment_color = "🔴"
        sentiment_class = "sentiment-negative"
    else:
        sentiment_color = "🟡"
        sentiment_class = "sentiment-neutral"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"{sentiment_color} **Sentiment:** {sentiment.title()}")
    with col2:
        st.markdown(f"🎯 **Confidence:** {confidence:.2f}")
    with col3:
        scores = result['sentiment_scores']
        st.markdown(f"📊 **Scores:** P:{scores.get('positive', 0):.2f} | N:{scores.get('negative', 0):.2f} | U:{scores.get('neutral', 0):.2f}")
    
    st.markdown("---")
    st.markdown("**📈 Stock Movement Prediction:**")
    
    movement = result['predicted_movement']
    pred_confidence = result['prediction_confidence']
    
    if movement == 'Up':
        movement_color = "📈"
        movement_class = "prediction-up"
    elif movement == 'Down':
        movement_color = "📉"
        movement_class = "prediction-down"
    else:
        movement_color = "➡️"
        movement_class = "prediction-flat"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"{movement_color} **Prediction:** {movement}")
    with col2:
        st.markdown(f"🎯 **Confidence:** {pred_confidence:.2f}")
    with col3:
        pred_scores = result['prediction_scores']
        st.markdown(f"📊 **Scores:** Up:{pred_scores.get('Up', 0):.2f} | Down:{pred_scores.get('Down', 0):.2f} | Flat:{pred_scores.get('Flat', 0):.2f}")
    
    st.markdown("---")
    st.markdown("**🏢 Detected Entities:**")
    
    entities = result['entities']
    organizations = entities.get('organizations', [])
    companies = entities.get('companies', [])
    
    if organizations or companies:
        col1, col2 = st.columns(2)
        with col1:
            if organizations:
                st.markdown("🏢 **Organizations:**")
                for org in organizations:
                    st.markdown(f"   • {org}")
        with col2:
            if companies:
                st.markdown("👤 **Companies/People:**")
                for company in companies:
                    st.markdown(f"   • {company}")
    else:
        st.markdown("*No specific entities detected*")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📈 AI News Sentiment Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Analyze financial news sentiment and predict stock movements using AI</p>', unsafe_allow_html=True)
    
    missing_keys = check_api_keys()
    if missing_keys:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("Please add the required API keys to your config_secrets.py file:")
        st.code("""
# In config_secrets.py:
NEWS_API_KEY = "your_news_api_key_here"
AZURE_TEXT_ANALYTICS_ENDPOINT = "https://your-resource.cognitiveservices.azure.com/"
AZURE_TEXT_ANALYTICS_KEY = "your_azure_text_analytics_key_here"
        """)
        return
    
    with st.sidebar:
        st.markdown('<h2 class="subheader">🔧 Configuration</h2>', unsafe_allow_html=True)
        
        query = st.text_input(
            "Enter stock/company name:",
            value=st.session_state.last_query or "AAPL",
            placeholder="e.g., AAPL, Tesla, Microsoft"
        )
        
        article_count = st.slider(
            "Number of articles to analyze:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        if st.button("🚀 Analyze News", type="primary"):
            if query.strip():
                st.session_state.last_query = query
                with st.spinner("Running complete analysis..."):
                    results = analyze_news(query.strip(), article_count)
                    st.session_state.analysis_results = results
                    st.rerun()
            else:
                st.error("Please enter a query")
        
        if st.button("🔄 Refresh Analysis"):
            if st.session_state.last_query:
                with st.spinner("Refreshing analysis..."):
                    results = analyze_news(st.session_state.last_query, article_count)
                    st.session_state.analysis_results = results
                    st.rerun()
            else:
                st.error("No previous query to refresh")
        
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size: 1.2rem; color: #2c3e50;">📊 About</h3>', unsafe_allow_html=True)
        st.markdown("""
        This dashboard uses:
        - **NewsAPI** for fetching financial news
        - **Azure Cognitive Services** for sentiment analysis and NER
        - **Custom ML model** for stock movement prediction
        """)
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        display_metrics(results)
        
        st.markdown('<h2 class="subheader">📊 Sentiment Analysis</h2>', unsafe_allow_html=True)
        display_sentiment_charts(results)
        
        st.markdown('<h2 class="subheader">📈 Stock Movement Predictions</h2>', unsafe_allow_html=True)
        display_prediction_chart(results)
        
        st.markdown('<h2 class="subheader">📰 Detailed Analysis</h2>', unsafe_allow_html=True)
        
        for result in results:
            display_article_card(result)
        
        st.markdown('<h2 class="subheader">💾 Export Results</h2>', unsafe_allow_html=True)
        if st.button("📥 Download Results as CSV"):
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
                    'Companies': ', '.join(result['entities']['companies']),
                    'URL': result['url']
                })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"news_analysis_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Use the sidebar to enter a stock/company name and start the analysis!")
        
        st.markdown('<h2 class="subheader">💡 Example Queries</h2>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Tech Companies**")
            st.markdown("- AAPL (Apple)")
            st.markdown("- MSFT (Microsoft)")
            st.markdown("- GOOGL (Google)")
        
        with col2:
            st.markdown("**Electric Vehicles**")
            st.markdown("- TSLA (Tesla)")
            st.markdown("- NIO (NIO)")
            st.markdown("- RIVN (Rivian)")
        
        with col3:
            st.markdown("**Financial Services**")
            st.markdown("- JPM (JPMorgan)")
            st.markdown("- BAC (Bank of America)")
            st.markdown("- GS (Goldman Sachs)")

if __name__ == "__main__":
    main()
