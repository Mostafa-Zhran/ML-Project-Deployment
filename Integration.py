import unittest
import os
import json
import numpy as np
from Model_Predicition import BreastCancerPredictor, train_and_save_model
import Modle  
from Modle import app  
import pandas as pd

class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create a test model if it doesn't exist
        cls.test_model_path = 'test_model.pkl'
        if not os.path.exists('breast_cancer_model.pkl'):
            train_and_save_model('Breast_cancer_dataset.csv')
        
        # Load test data
        cls.test_data = pd.read_csv('Breast_cancer_dataset.csv')
        
        # Initialize Flask test client
        app.config['TESTING'] = True
        cls.client = app.test_client()
    
    def test_1_model_loading(self):
        """Test if model loads correctly"""
        predictor = BreastCancerPredictor.load_model('breast_cancer_model.pkl')
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.scaler)
        self.assertEqual(len(predictor.feature_names), 30)
    
    def test_2_predictor_initialization(self):
        """Test predictor initialization with data"""
        predictor = BreastCancerPredictor('Breast_cancer_dataset.csv')
        self.assertIsNotNone(predictor.model)
    
    def test_3_prediction_shape(self):
        """Test prediction output shape"""
        predictor = BreastCancerPredictor.load_model('breast_cancer_model.pkl')
        sample = self.test_data.iloc[0].drop(['id', 'diagnosis', 'Unnamed: 32'], errors='ignore')
        result = predictor.predict(sample.values)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
    
    def test_4_api_prediction(self):
        """Test prediction through API endpoint"""
        sample = self.test_data.iloc[0].to_dict()
        
        # Convert all values to float
        features = {}
        for k, v in sample.items():
            if k not in ['id', 'diagnosis', 'Unnamed: 32']:
                try:
                    features[k] = float(v)
                except (ValueError, TypeError):
                    pass
        
        response = self.client.post('/predict', 
                                 data=json.dumps(features),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertIn('probabilities', data)
    
    def test_5_missing_values(self):
        """Test prediction with missing values"""
        sample = {}
        for i in range(30):
            sample[f'feature_{i}'] = 0.0
        
        response = self.client.post('/predict', 
                                 data=json.dumps(sample),
                                 content_type='application/json')
        
        # Should still return 200 with default values
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
