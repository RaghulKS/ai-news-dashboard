"""
Stock Prediction Model Training

This module trains a machine learning classifier to predict stock movement
(up/down/flat) based on sentiment analysis and text features from news headlines.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictorTrainer:
    """
    A class to train and evaluate stock prediction models using sentiment and text features.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the Stock Predictor Trainer.
        
        Args:
            model_path (str): Path to save/load models
        """
        self.model_path = model_path or os.path.join('ml', 'models')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_names = []
        
    def create_mock_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Create a mock dataset for training the stock prediction model.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Mock dataset with features and labels
        """
        logger.info(f"Creating mock dataset with {num_samples} samples...")
        
        # Comprehensive sample headlines with different sentiments and companies
        positive_headlines = [
            "Apple stock surges to new record high after strong earnings",
            "Microsoft reports record profits, shares jump 8%",
            "Tesla beats quarterly expectations, stock rallies 15%",
            "Amazon revenue soars, stock hits all-time high",
            "Google parent Alphabet reports stellar earnings",
            "Netflix subscriber growth exceeds expectations",
            "Bank of America reports strong quarterly results",
            "Goldman Sachs beats earnings estimates",
            "JPMorgan Chase posts record profits",
            "Walmart stock rises on positive outlook",
            "Meta Platforms stock soars after strong ad revenue",
            "NVIDIA reports record GPU sales, stock jumps 12%",
            "Salesforce beats revenue expectations",
            "Adobe reports strong subscription growth",
            "Oracle posts better-than-expected earnings",
            "Intel stock rises on new chip technology",
            "AMD beats earnings estimates, stock rallies",
            "Cisco reports strong quarterly results",
            "IBM posts positive earnings surprise",
            "Verizon stock gains on strong wireless growth"
        ]
        
        negative_headlines = [
            "Apple stock plummets after disappointing earnings",
            "Microsoft shares fall on weak guidance",
            "Tesla faces regulatory scrutiny, stock drops 10%",
            "Amazon misses revenue targets, shares decline",
            "Google parent Alphabet reports lower than expected earnings",
            "Netflix loses subscribers, stock crashes 25%",
            "Bank of America reports losses in trading division",
            "Goldman Sachs misses earnings expectations",
            "JPMorgan Chase faces regulatory fines",
            "Walmart stock falls on poor quarterly results",
            "Meta Platforms stock drops on privacy concerns",
            "NVIDIA stock falls on chip shortage",
            "Salesforce reports disappointing growth",
            "Adobe stock declines on subscription slowdown",
            "Oracle misses revenue expectations",
            "Intel stock drops on manufacturing delays",
            "AMD stock falls on supply chain issues",
            "Cisco reports weak quarterly results",
            "IBM stock declines on cloud competition",
            "Verizon stock drops on subscriber losses"
        ]
        
        neutral_headlines = [
            "Apple announces new product line",
            "Microsoft releases quarterly earnings report",
            "Tesla announces new factory location",
            "Amazon expands into new market",
            "Google parent Alphabet holds annual meeting",
            "Netflix announces content partnership",
            "Bank of America appoints new executive",
            "Goldman Sachs reports quarterly results",
            "JPMorgan Chase announces restructuring",
            "Walmart opens new store locations",
            "Meta Platforms updates privacy policy",
            "NVIDIA announces new chip architecture",
            "Salesforce acquires new company",
            "Adobe releases software update",
            "Oracle announces cloud partnership",
            "Intel unveils new processor line",
            "AMD announces new graphics cards",
            "Cisco expands network solutions",
            "IBM launches new AI services",
            "Verizon expands 5G network"
        ]
        
        # News sources for realism
        news_sources = [
            "Reuters", "Bloomberg", "CNBC", "MarketWatch", "Yahoo Finance",
            "Financial Times", "Wall Street Journal", "Barron's", "Investor's Business Daily",
            "Seeking Alpha", "Motley Fool", "Benzinga", "TheStreet", "Zacks"
        ]
        
        # Companies for entity recognition
        companies = [
            "Apple Inc.", "Microsoft Corporation", "Tesla Inc.", "Amazon.com Inc.",
            "Alphabet Inc.", "Netflix Inc.", "Bank of America", "Goldman Sachs Group",
            "JPMorgan Chase", "Walmart Inc.", "Meta Platforms", "NVIDIA Corporation",
            "Salesforce Inc.", "Adobe Inc.", "Oracle Corporation", "Intel Corporation",
            "Advanced Micro Devices", "Cisco Systems", "IBM Corporation", "Verizon Communications"
        ]
        
        # Generate mock data
        data = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(num_samples):
            # Randomly select sentiment and headline
            sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.4, 0.2])
            
            if sentiment == 'positive':
                headline = np.random.choice(positive_headlines)
                stock_movement = np.random.choice(['up', 'flat'], p=[0.7, 0.3])
            elif sentiment == 'negative':
                headline = np.random.choice(negative_headlines)
                stock_movement = np.random.choice(['down', 'flat'], p=[0.7, 0.3])
            else:  # neutral
                headline = np.random.choice(neutral_headlines)
                stock_movement = np.random.choice(['flat', 'up', 'down'], p=[0.6, 0.2, 0.2])
            
            # Generate realistic sentiment scores
            if sentiment == 'positive':
                sentiment_score = np.random.uniform(0.65, 0.95)
                positive_conf = np.random.uniform(0.6, 0.9)
                negative_conf = np.random.uniform(0.05, 0.25)
                neutral_conf = np.random.uniform(0.05, 0.25)
            elif sentiment == 'negative':
                sentiment_score = np.random.uniform(0.05, 0.35)
                positive_conf = np.random.uniform(0.05, 0.25)
                negative_conf = np.random.uniform(0.6, 0.9)
                neutral_conf = np.random.uniform(0.05, 0.25)
            else:  # neutral
                sentiment_score = np.random.uniform(0.35, 0.65)
                positive_conf = np.random.uniform(0.2, 0.4)
                negative_conf = np.random.uniform(0.2, 0.4)
                neutral_conf = np.random.uniform(0.4, 0.7)
            
            # Normalize confidence scores
            total_conf = positive_conf + negative_conf + neutral_conf
            positive_conf /= total_conf
            negative_conf /= total_conf
            neutral_conf /= total_conf
            
            # Generate realistic market data
            volume = np.random.randint(1000000, 50000000)
            price_change = np.random.uniform(-0.15, 0.15)
            market_cap = np.random.uniform(1e9, 2e12)
            
            # Generate date (more recent for realism)
            date = datetime.now() - timedelta(days=np.random.randint(1, 90))
            
            # Select random source and company
            source = np.random.choice(news_sources)
            company = np.random.choice(companies)
            
            # Generate URL
            url = f"https://{source.lower().replace(' ', '').replace('.', '')}.com/article/{np.random.randint(10000, 99999)}"
            
            # Create rich sample with detailed metadata
            sample = {
                'date': date.strftime('%Y-%m-%d'),
                'headline': headline,
                'description': f"{headline} according to {source} analysis.",
                'content': f"{headline}. The company reported significant changes in their quarterly performance. Analysts expect this to impact the stock price in the coming weeks.",
                'url': url,
                'source': source,
                'published_at': date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'positive_confidence': positive_conf,
                'negative_confidence': negative_conf,
                'neutral_confidence': neutral_conf,
                'stock_movement': stock_movement,
                'volume': volume,
                'price_change': price_change,
                'market_cap': market_cap,
                'company': company,
                'query': np.random.choice(['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'NFLX', 'META', 'NVDA'])
            }
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        logger.info(f"Created mock dataset with {len(df)} samples")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from the dataset for model training.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        logger.info("Extracting features from dataset...")
        
        # Text features
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Fit and transform text features
        text_features = vectorizer.fit_transform(df['headline'])
        self.vectorizer = vectorizer
        self.feature_names = vectorizer.get_feature_names_out()
        
        # Numerical features
        numerical_features = df[[
            'sentiment_score',
            'positive_confidence',
            'negative_confidence',
            'neutral_confidence',
            'volume',
            'price_change',
            'market_cap'
        ]].values
        
        # Combine features
        features = np.hstack([text_features.toarray(), numerical_features])
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['stock_movement'])
        self.label_encoder = label_encoder
        
        logger.info(f"Extracted {features.shape[1]} features for {features.shape[0]} samples")
        return features, labels
    
    def train_model(self, features: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.
        
        Args:
            features (np.ndarray): Training features
            labels (np.ndarray): Training labels
            
        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("Training Random Forest classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, features, labels, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:]
        
        logger.info("Top 10 most important features:")
        for i, idx in enumerate(reversed(top_features)):
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
            else:
                feature_name = f"numerical_feature_{idx - len(self.feature_names)}"
            logger.info(f"  {feature_name}: {feature_importance[idx]:.3f}")
        
        self.model = model
        return model
    
    def save_model(self, filename: str = 'stock_predictor.joblib') -> str:
        """
        Save the trained model and components.
        
        Args:
            filename (str): Name of the file to save
            
        Returns:
            str: Path to saved model file
        """
        if not self.model:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.model_path, filename)
        joblib.dump(model_data, filepath)
        
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filename: str = 'stock_predictor.joblib') -> bool:
        """
        Load a trained model and components.
        
        Args:
            filename (str): Name of the file to load
            
        Returns:
            bool: True if loaded successfully
        """
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
            
            logger.info(f"Model loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, headlines: List[str], sentiment_scores: List[Dict]) -> List[Dict]:
        """
        Predict stock movement for given headlines and sentiment data.
        
        Args:
            headlines (List[str]): List of news headlines
            sentiment_scores (List[Dict]): List of sentiment analysis results
            
        Returns:
            List[Dict]: Predictions with confidence scores
        """
        if not self.model:
            raise ValueError("No trained model available")
        
        # Prepare features
        features_list = []
        
        for headline, sentiment_data in zip(headlines, sentiment_scores):
            # Text features
            text_features = self.vectorizer.transform([headline]).toarray()
            
            # Numerical features
            sentiment_score = sentiment_data.get('sentiment_score', 0.5)
            confidence_scores = sentiment_data.get('confidence_scores', {})
            
            numerical_features = np.array([[
                sentiment_score,
                confidence_scores.get('positive', 0.33),
                confidence_scores.get('negative', 0.33),
                confidence_scores.get('neutral', 0.34),
                0,  # volume (placeholder)
                0,  # price_change (placeholder)
                0   # market_cap (placeholder)
            ]])
            
            # Combine features
            features = np.hstack([text_features, numerical_features])
            features_list.append(features[0])
        
        features_array = np.array(features_list)
        
        # Make predictions
        predictions = self.model.predict(features_array)
        probabilities = self.model.predict_proba(features_array)
        
        # Convert predictions back to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Prepare results
        results = []
        for i, (headline, label, prob) in enumerate(zip(headlines, predicted_labels, probabilities)):
            result = {
                'headline': headline,
                'predicted_movement': label,
                'confidence_scores': {
                    'up': prob[0],
                    'down': prob[1],
                    'flat': prob[2]
                },
                'max_confidence': max(prob)
            }
            results.append(result)
        
        return results


def main():
    """
    Main function to train the stock prediction model.
    """
    try:
        # Initialize trainer
        trainer = StockPredictorTrainer()
        
        # Create mock dataset
        print("📊 Creating mock dataset...")
        df = trainer.create_mock_dataset(num_samples=1000)
        
        # Save mock dataset
        os.makedirs('data', exist_ok=True)
        csv_path = os.path.join('data', 'historical_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Mock dataset saved to: {csv_path}")
        
        # Extract features
        print("\n🔧 Extracting features...")
        features, labels = trainer.extract_features(df)
        
        # Train model
        print("\n🤖 Training Random Forest model...")
        model = trainer.train_model(features, labels)
        
        # Save model
        print("\n💾 Saving model...")
        model_path = trainer.save_model()
        print(f"✅ Model saved to: {model_path}")
        
        # Test predictions
        print("\n🧪 Testing predictions...")
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
        
        predictions = trainer.predict(test_headlines, test_sentiments)
        
        print("\n📈 Prediction Results:")
        for pred in predictions:
            movement = pred['predicted_movement']
            confidence = pred['max_confidence']
            emoji = "📈" if movement == "up" else "📉" if movement == "down" else "➡️"
            print(f"{emoji} {pred['headline']}")
            print(f"   Prediction: {movement} (confidence: {confidence:.2f})")
        
        print("\n🎉 Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 