"""
Test script for the Stock Predictor Model Interface.
Run this to test the model.py interface functionality.
"""

import os
import sys
from ml.model import StockPredictor

def test_model_interface():
    """Test the Stock Predictor model interface."""
    
    print("🤖 Testing Stock Predictor Model Interface")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = StockPredictor()
        
        # Check if model is available
        if not predictor.is_model_available():
            print("❌ Model not available. Please train the model first:")
            print("   python ml/train_model.py")
            return False
        
        print("✅ Stock Predictor initialized successfully")
        
        # Test model info
        print("\n📋 Model Information:")
        model_info = predictor.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Test single prediction
        print("\n🧪 Testing single prediction...")
        test_headline = "Apple stock surges to new record high after strong earnings"
        test_sentiment = {
            'sentiment': 'positive',
            'confidence_scores': {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1},
            'sentiment_score': 0.8
        }
        
        result = predictor.predict_single(test_headline, test_sentiment)
        
        print(f"Headline: {result['headline']}")
        print(f"Prediction: {result['predicted_movement']}")
        print(f"Confidence: {result['max_confidence']:.2f}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence breakdown:")
        for movement, conf in result['confidence_scores'].items():
            emoji = "📈" if movement == "Up" else "📉" if movement == "Down" else "➡️"
            print(f"   {emoji} {movement}: {conf:.2f}")
        
        # Test batch prediction
        print("\n📊 Testing batch prediction...")
        test_headlines = [
            "Apple stock surges to new record high after strong earnings",
            "Tesla faces regulatory scrutiny, stock drops",
            "Microsoft announces new product line",
            "Amazon beats earnings expectations, shares rally",
            "Netflix loses subscribers, stock crashes"
        ]
        
        test_sentiments = [
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
            },
            {
                'sentiment': 'positive',
                'confidence_scores': {'positive': 0.7, 'negative': 0.2, 'neutral': 0.1},
                'sentiment_score': 0.7
            },
            {
                'sentiment': 'negative',
                'confidence_scores': {'positive': 0.2, 'negative': 0.7, 'neutral': 0.1},
                'sentiment_score': 0.3
            }
        ]
        
        batch_results = predictor.predict_batch(test_headlines, test_sentiments)
        
        print("\n📈 Batch Prediction Results:")
        print("-" * 50)
        
        for i, result in enumerate(batch_results, 1):
            movement = result['predicted_movement']
            confidence = result['max_confidence']
            sentiment = result['sentiment']
            
            # Emoji for movement
            if movement == "Up":
                emoji = "📈"
            elif movement == "Down":
                emoji = "📉"
            else:
                emoji = "➡️"
            
            print(f"\n{i}. {emoji} {result['headline']}")
            print(f"   Prediction: {movement}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Sentiment: {sentiment}")
        
        # Test with article format
        print("\n📰 Testing with article format...")
        test_articles = [
            {
                'title': 'Apple reports record-breaking quarterly earnings',
                'sentiment_analysis': {
                    'sentiment': 'positive',
                    'confidence_scores': {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1},
                    'sentiment_score': 0.8
                }
            },
            {
                'title': 'Tesla faces multiple lawsuits over safety concerns',
                'sentiment_analysis': {
                    'sentiment': 'negative',
                    'confidence_scores': {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1},
                    'sentiment_score': 0.2
                }
            },
            {
                'title': 'Microsoft announces strategic partnership',
                'sentiment_analysis': {
                    'sentiment': 'neutral',
                    'confidence_scores': {'positive': 0.3, 'negative': 0.2, 'neutral': 0.5},
                    'sentiment_score': 0.5
                }
            }
        ]
        
        article_results = predictor.predict_from_articles(test_articles)
        
        print("\n📊 Article Prediction Results:")
        for i, result in enumerate(article_results, 1):
            movement = result['predicted_movement']
            confidence = result['max_confidence']
            emoji = "📈" if movement == "Up" else "📉" if movement == "Down" else "➡️"
            print(f"{i}. {emoji} {result['headline']} -> {movement} ({confidence:.2f})")
        
        # Test edge cases
        print("\n🔍 Testing edge cases...")
        
        # Empty headline
        try:
            empty_result = predictor.predict_single("", test_sentiment)
            print(f"Empty headline result: {empty_result['predicted_movement']}")
        except Exception as e:
            print(f"Empty headline error: {e}")
        
        # Missing sentiment data
        try:
            minimal_sentiment = {'sentiment': 'positive'}
            minimal_result = predictor.predict_single(test_headline, minimal_sentiment)
            print(f"Minimal sentiment result: {minimal_result['predicted_movement']}")
        except Exception as e:
            print(f"Minimal sentiment error: {e}")
        
        # Test prediction summary
        print("\n📊 Prediction Summary:")
        all_results = batch_results + article_results
        movement_counts = {}
        total_confidence = 0
        
        for result in all_results:
            movement = result['predicted_movement']
            movement_counts[movement] = movement_counts.get(movement, 0) + 1
            total_confidence += result['max_confidence']
        
        print(f"Total predictions: {len(all_results)}")
        for movement, count in movement_counts.items():
            emoji = "📈" if movement == "Up" else "📉" if movement == "Down" else "➡️"
            percentage = (count / len(all_results)) * 100
            print(f"   {emoji} {movement}: {count} ({percentage:.1f}%)")
        
        avg_confidence = total_confidence / len(all_results) if all_results else 0
        print(f"Average confidence: {avg_confidence:.2f}")
        
        print("\n🎉 Model interface test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model interface testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading functionality."""
    
    print("\n📂 Testing Model Loading")
    print("=" * 30)
    
    try:
        # Test with default path
        predictor1 = StockPredictor()
        if predictor1.is_model_available():
            print("✅ Model loaded with default path")
        else:
            print("❌ Model not found with default path")
        
        # Test with explicit path
        model_path = os.path.join('ml', 'models', 'stock_predictor.joblib')
        if os.path.exists(model_path):
            predictor2 = StockPredictor(model_path)
            if predictor2.is_model_available():
                print("✅ Model loaded with explicit path")
            else:
                print("❌ Model loading failed with explicit path")
        else:
            print("❌ Model file not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model loading test: {e}")
        return False

if __name__ == "__main__":
    # Run main test
    success1 = test_model_interface()
    
    # Run model loading test
    success2 = test_model_loading()
    
    if success1 and success2:
        print("\n🎉 All model interface tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 