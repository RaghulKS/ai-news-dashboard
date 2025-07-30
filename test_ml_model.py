"""
Test script for ML Model Training and Prediction.
Run this to test the stock prediction model functionality.
"""

import os
import sys
import pandas as pd
from ml.train_model import StockPredictorTrainer

def test_ml_model():
    """Test the ML model training and prediction functionality."""
    
    print("🤖 Testing ML Model Training and Prediction")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = StockPredictorTrainer()
        print("✅ StockPredictorTrainer initialized")
        
        # Test mock dataset creation
        print("\n📊 Testing mock dataset creation...")
        df = trainer.create_mock_dataset(num_samples=500)  # Smaller dataset for testing
        
        print(f"✅ Created dataset with {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        print(f"   Stock movement distribution: {df['stock_movement'].value_counts().to_dict()}")
        
        # Save dataset
        os.makedirs('data', exist_ok=True)
        csv_path = os.path.join('data', 'historical_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Dataset saved to: {csv_path}")
        
        # Test feature extraction
        print("\n🔧 Testing feature extraction...")
        features, labels = trainer.extract_features(df)
        print(f"✅ Extracted {features.shape[1]} features for {features.shape[0]} samples")
        print(f"   Feature shape: {features.shape}")
        print(f"   Label shape: {labels.shape}")
        
        # Test model training
        print("\n🤖 Testing model training...")
        model = trainer.train_model(features, labels)
        print("✅ Model training completed")
        
        # Test model saving
        print("\n💾 Testing model saving...")
        model_path = trainer.save_model()
        print(f"✅ Model saved to: {model_path}")
        
        # Test model loading
        print("\n📂 Testing model loading...")
        new_trainer = StockPredictorTrainer()
        success = new_trainer.load_model()
        if success:
            print("✅ Model loaded successfully")
        else:
            print("❌ Model loading failed")
            return False
        
        # Test predictions
        print("\n🧪 Testing predictions...")
        test_headlines = [
            "Apple stock surges to new record high after strong earnings",
            "Tesla faces regulatory scrutiny, stock drops",
            "Microsoft announces new product line",
            "Amazon beats earnings expectations, shares rally",
            "Netflix loses subscribers, stock crashes"
        ]
        
        test_sentiments = [
            {
                'sentiment_score': 0.8,
                'confidence_scores': {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}
            },
            {
                'sentiment_score': 0.2,
                'confidence_scores': {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1}
            },
            {
                'sentiment_score': 0.5,
                'confidence_scores': {'positive': 0.3, 'negative': 0.2, 'neutral': 0.5}
            },
            {
                'sentiment_score': 0.7,
                'confidence_scores': {'positive': 0.7, 'negative': 0.2, 'neutral': 0.1}
            },
            {
                'sentiment_score': 0.3,
                'confidence_scores': {'positive': 0.2, 'negative': 0.7, 'neutral': 0.1}
            }
        ]
        
        predictions = new_trainer.predict(test_headlines, test_sentiments)
        
        print("\n📈 Prediction Results:")
        print("-" * 50)
        
        for i, pred in enumerate(predictions, 1):
            headline = pred['headline']
            movement = pred['predicted_movement']
            confidence = pred['max_confidence']
            confidence_scores = pred['confidence_scores']
            
            # Emoji for movement
            if movement == "up":
                emoji = "📈"
            elif movement == "down":
                emoji = "📉"
            else:
                emoji = "➡️"
            
            print(f"\n{i}. {emoji} {headline}")
            print(f"   Prediction: {movement.upper()}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Confidence breakdown:")
            print(f"     📈 Up: {confidence_scores['up']:.2f}")
            print(f"     📉 Down: {confidence_scores['down']:.2f}")
            print(f"     ➡️ Flat: {confidence_scores['flat']:.2f}")
        
        # Test with real sentiment data format
        print("\n" + "=" * 50)
        print("🎭 Testing with real sentiment data format...")
        
        # Simulate real sentiment analysis results
        real_sentiment_data = [
            {
                'sentiment': 'positive',
                'confidence_scores': {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1},
                'sentiment_score': 0.8
            },
            {
                'sentiment': 'negative',
                'confidence_scores': {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1},
                'sentiment_score': 0.2
            },
            {
                'sentiment': 'neutral',
                'confidence_scores': {'positive': 0.3, 'negative': 0.2, 'neutral': 0.5},
                'sentiment_score': 0.5
            }
        ]
        
        real_headlines = [
            "Apple reports record-breaking quarterly earnings",
            "Tesla faces multiple lawsuits over safety concerns",
            "Microsoft announces strategic partnership"
        ]
        
        real_predictions = new_trainer.predict(real_headlines, real_sentiment_data)
        
        print("\n📊 Real Data Predictions:")
        for pred in real_predictions:
            movement = pred['predicted_movement']
            confidence = pred['max_confidence']
            emoji = "📈" if movement == "up" else "📉" if movement == "down" else "➡️"
            print(f"{emoji} {pred['headline']} -> {movement.upper()} ({confidence:.2f})")
        
        # Test model performance metrics
        print("\n" + "=" * 50)
        print("📊 Model Performance Summary:")
        print(f"✅ Dataset size: {len(df)} samples")
        print(f"✅ Features extracted: {features.shape[1]}")
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Predictions tested: {len(predictions)} samples")
        
        print("\n🎉 All ML model tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during ML model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_quality():
    """Test the quality of the generated mock dataset."""
    
    print("\n🔍 Testing Dataset Quality")
    print("=" * 30)
    
    try:
        # Load the dataset
        csv_path = os.path.join('data', 'historical_data.csv')
        if not os.path.exists(csv_path):
            print("❌ Dataset file not found. Run the main test first.")
            return False
        
        df = pd.read_csv(csv_path)
        
        print(f"✅ Dataset loaded: {len(df)} samples")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Check data quality
        print(f"\n📊 Data Quality Check:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicate rows: {df.duplicated().sum()}")
        
        # Check distributions
        print(f"\n📈 Sentiment Distribution:")
        sentiment_dist = df['sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        print(f"\n📊 Stock Movement Distribution:")
        movement_dist = df['stock_movement'].value_counts()
        for movement, count in movement_dist.items():
            percentage = (count / len(df)) * 100
            emoji = "📈" if movement == "up" else "📉" if movement == "down" else "➡️"
            print(f"   {emoji} {movement}: {count} ({percentage:.1f}%)")
        
        # Check feature ranges
        print(f"\n🔢 Feature Statistics:")
        numerical_cols = ['sentiment_score', 'positive_confidence', 'negative_confidence', 
                         'neutral_confidence', 'volume', 'price_change', 'market_cap']
        
        for col in numerical_cols:
            if col in df.columns:
                print(f"   {col}: {df[col].min():.3f} to {df[col].max():.3f} (mean: {df[col].mean():.3f})")
        
        print("✅ Dataset quality check completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during dataset quality check: {e}")
        return False

if __name__ == "__main__":
    # Run main test
    success1 = test_ml_model()
    
    # Run dataset quality test
    success2 = test_dataset_quality()
    
    if success1 and success2:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 