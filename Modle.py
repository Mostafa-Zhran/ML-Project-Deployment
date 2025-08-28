from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os
from waitress import serve
from Model_Predicition import BreastCancerPredictor

# Initialize predictor
try:
    predictor = BreastCancerPredictor.load_model('breast_cancer_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configuration
app.config.update(
    DEBUG=False,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
    JSON_SORT_KEYS=False
)

# Load the trained model and scaler
model_path = 'best_model_logistic_regression.pkl'
try:
    print(f"\n=== Loading model from: {os.path.abspath(model_path)} ===")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    # Extract model and scaler from the dictionary
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    
    if model is None or scaler is None:
        raise ValueError("Model or scaler not found in the model file")
        
    # Print model information
    print(f"Model type: {type(model).__name__}")
    print(f"Scaler type: {type(scaler).__name__}")
    print("Model and scaler loaded successfully!")
    print("==================================\n")
except Exception as e:
    print(f"\n=== Error loading model ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {str(e)}")
    print("======================\n")
    raise

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "message": "The requested resource was not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "message": "An unexpected error occurred"}), 500

# API Routes
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load predictor if not already loaded
        global predictor
        if predictor is None:
            try:
                predictor = BreastCancerPredictor.load_model('breast_cancer_model.pkl')
            except Exception as e:
                return jsonify({"error": "Failed to load model", "details": str(e)}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Make prediction using the predictor
        result = predictor.predict([
            data.get('radius_mean', 0),
            data.get('texture_mean', 0),
            data.get('perimeter_mean', 0),
            data.get('area_mean', 0),
            data.get('smoothness_mean', 0),
            data.get('compactness_mean', 0),
            data.get('concavity_mean', 0),
            data.get('concave_points_mean', 0),
            data.get('symmetry_mean', 0),
            data.get('fractal_dimension_mean', 0),
            data.get('radius_se', 0),
            data.get('texture_se', 0),
            data.get('perimeter_se', 0),
            data.get('area_se', 0),
            data.get('smoothness_se', 0),
            data.get('compactness_se', 0),
            data.get('concavity_se', 0),
            data.get('concave_points_se', 0),
            data.get('symmetry_se', 0),
            data.get('fractal_dimension_se', 0),
            data.get('radius_worst', 0),
            data.get('texture_worst', 0),
            data.get('perimeter_worst', 0),
            data.get('area_worst', 0),
            data.get('smoothness_worst', 0),
            data.get('compactness_worst', 0),
            data.get('concavity_worst', 0),
            data.get('concave_points_worst', 0),
            data.get('symmetry_worst', 0),
            data.get('fractal_dimension_worst', 0)
        ])
        
        return jsonify(result)
            
    except (ValueError, TypeError) as e:
        return jsonify({"error": "Invalid input data", "details": str(e)}), 400
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n=== Prediction Error ===")
        print(error_details)
        print("======================\n")
        app.logger.error(f"Prediction error: {str(e)}\n{error_details}")
        return jsonify({
            "error": "Error making prediction",
            "details": str(e),
            "type": type(e).__name__
        }), 500
        return jsonify({"error": "An unexpected error occurred"}), 500

# Serve static files from the current directory
import os

# Serve index.html for the root URL
@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

# Serve other static files (CSS, JS, etc.)
@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)

if __name__ == "__main__":
    # For development
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True, host='localhost', port=5000)
    else:
        # For production
        print("Starting production server on http://localhost:8080")
        serve(app, host='localhost', port=8080, threads=4, url_scheme='http')
