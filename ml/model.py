import os
import logging
import joblib
import numpy as np
from typing import List, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    def __init__(self, model_path: str = 'ml/models'):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        
        os.makedirs(model_path, exist_ok=True)

    def is_model_available(self) -> bool:
        model_file = os.path.join(self.model_path, 'stock_predictor.joblib')
        return os.path.exists(model_file) and self.is_loaded

    def predict_from_articles(self, articles: List[Dict]) -> List[Dict]:
        headlines = []
        sentiment_data = []
        
        for article in articles:
            headline = article.get('title', '')
            sentiment_analysis = article.get('sentiment_analysis', {})
            
            if headline and sentiment_analysis:
                headlines.append(headline)
                sentiment_data.append(sentiment_analysis)
        
        if not self.is_loaded:
            logger.warning("Model not loaded, using sentiment-based fallback predictions")
            return self._fallback_predictions(headlines, sentiment_data)
        
        return self.predict_batch(headlines, sentiment_data)
    
    def _fallback_predictions(self, headlines: List[str], sentiment_data: List[Dict]) -> List[Dict]:
        results = []
        
        for headline, sentiment in zip(headlines, sentiment_data):
            sentiment_type = sentiment.get('sentiment', 'neutral')
            confidence_scores = sentiment.get('confidence_scores', {})
            
            if sentiment_type == 'positive':
                predicted_movement = 'Up'
                confidence = confidence_scores.get('positive', 0.6)
                pred_scores = {'Up': confidence, 'Down': 0.2, 'Flat': 1-confidence-0.2}
            elif sentiment_type == 'negative':
                predicted_movement = 'Down'
                confidence = confidence_scores.get('negative', 0.6)
                pred_scores = {'Up': 0.2, 'Down': confidence, 'Flat': 1-confidence-0.2}
            else:
                predicted_movement = 'Flat'
                confidence = confidence_scores.get('neutral', 0.5)
                pred_scores = {'Up': 0.25, 'Down': 0.25, 'Flat': confidence}
            
            total = sum(pred_scores.values())
            pred_scores = {k: v/total for k, v in pred_scores.items()}
            
            result = {
                'headline': headline,
                'predicted_movement': predicted_movement,
                'confidence_scores': pred_scores,
                'max_confidence': max(pred_scores.values()),
                'sentiment': sentiment_type,
                'method': 'sentiment_fallback'
            }
            
            results.append(result)
        
        logger.info(f"Generated {len(results)} fallback predictions based on sentiment")
        return results
    
    def _prepare_features(self, headlines: List[str], sentiment_data: List[Dict]) -> np.ndarray:
        if not self.vectorizer:
            raise ValueError("Model not trained. Please train the model first.")
        
        text_features = self.vectorizer.transform(headlines).toarray()
        
        numerical_features = []
        for sentiment in sentiment_data:
            sentiment_score = sentiment.get('sentiment_score', 0.5)
            confidence_scores = sentiment.get('confidence_scores', {})
            
            numerical_features.append([
                sentiment_score,
                confidence_scores.get('positive', 0.33),
                confidence_scores.get('negative', 0.33),
                confidence_scores.get('neutral', 0.34),
                0, 0, 0
            ])
        
        numerical_features = np.array(numerical_features)
        features = np.hstack([text_features, numerical_features])
        
        return features

    def predict_batch(self, headlines: List[str], sentiment_data: List[Dict]) -> List[Dict]:
        if not self.model:
            raise ValueError("No trained model available")
        
        features = self._prepare_features(headlines, sentiment_data)
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        results = []
        for i, (headline, label, prob) in enumerate(zip(headlines, predicted_labels, probabilities)):
            result = {
                'headline': headline,
                'predicted_movement': label,
                'confidence_scores': {
                    'Up': prob[0],
                    'Down': prob[1],
                    'Flat': prob[2]
                },
                'max_confidence': max(prob),
                'method': 'ml_model'
            }
            results.append(result)
        
        return results

    def load_model(self, filename: str = 'stock_predictor.joblib') -> bool:
        filepath = os.path.join(self.model_path, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            
            self.is_loaded = True
            logger.info(f"Model loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


def main():
    try:
        predictor = StockPredictor()
        
        if predictor.load_model():
            print("✅ Model loaded successfully")
            
            test_headlines = [
                "Apple stock surges to new record high after strong earnings",
                "Tesla faces regulatory scrutiny, stock drops",
                "Microsoft announces new product line"
            ]
            
            test_sentiments = [
                {'sentiment_score': 0.8, 'confidence_scores': {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}},
                {'sentiment_score': 0.2, 'confidence_scores': {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1}},
                {'sentiment_score': 0.5, 'confidence_scores': {'positive': 0.3, 'negative': 0.2, 'neutral': 0.5}}
            ]
            
            predictions = predictor.predict_batch(test_headlines, test_sentiments)
            
            print("\n📈 Prediction Results:")
            for pred in predictions:
                movement = pred['predicted_movement']
                confidence = pred['max_confidence']
                emoji = "📈" if movement == "Up" else "📉" if movement == "Down" else "➡️"
                print(f"{emoji} {pred['headline']}")
                print(f"   Prediction: {movement} (confidence: {confidence:.2f})")
        else:
            print("❌ Model not available, using fallback predictions")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 