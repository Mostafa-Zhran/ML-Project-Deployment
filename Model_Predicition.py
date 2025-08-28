from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle
import os

class BreastCancerPredictor:
    def __init__(self, data_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        if data_path:
            X_train, X_test, y_train, y_test = self.load_and_preprocess_data(data_path)
            self.train_model(X_train, y_train)
    
    def load_and_preprocess_data(self, data_path):
        # Load data
        data = pd.read_csv(data_path)
        
        # Drop unnecessary columns
        data = data.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
        
        # Encode target variable
        le = LabelEncoder()
        data['diagnosis'] = le.fit_transform(data['diagnosis'])
        
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Split features and target
        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        # Train logistic regression model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, features):
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not initialized. Please train the model first.")
        
        # Convert features to numpy array and reshape
        features_array = np.array(features, dtype=float).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        return {
            'prediction': int(prediction),
            'confidence': float(max(probabilities)),
            'probabilities': {
                'benign': float(probabilities[0]),
                'malignant': float(probabilities[1])
            }
        }
    
    def save_model(self, filename):
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not initialized. Nothing to save.")
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
    
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            saved_data = pickle.load(f)
        
        predictor = BreastCancerPredictor()
        predictor.model = saved_data['model']
        predictor.scaler = saved_data['scaler']
        predictor.feature_names = saved_data.get('feature_names', [])
        
        return predictor

def train_and_save_model(data_path, output_file='breast_cancer_model.pkl'):
    """
    Train a new model and save it to disk
    """
    try:
        print(f"Training new model with data from {data_path}")
        
        # Initialize predictor and load data
        predictor = BreastCancerPredictor()
        X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data(data_path)
        
        # Train model
        print("Training model...")
        predictor.train_model(X_train, y_train)
        
        # Evaluate model
        train_acc = predictor.model.score(X_train, y_train)
        test_acc = predictor.model.score(X_test, y_test)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Save model
        predictor.save_model(output_file)
        print(f"Model saved to {output_file}")
        
        return predictor
    except Exception as e:
        print(f"Error in train_and_save_model: {str(e)}")
        raise
