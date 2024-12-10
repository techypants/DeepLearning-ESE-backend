import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load or Train Model and Data
def load_or_train_model():
    try:
        # Attempt to load the model from a pickle file
        with open('real_estate_model_full.pkl', 'rb') as f:
            model_data = pickle.load(f)

        print("Model loaded successfully!")
        return model_data
    except FileNotFoundError:
        print("Model file not found. Please train and pickle the model first.")
        return None
def validate_and_preprocess_data(data):
    try:
        # Ensure numerical fields are present and valid
        for key in ['Total_Area', 'Price_per_SQFT', 'Baths']:
            if key not in data:
                raise ValueError(f"Missing required field: {key}")
            if not isinstance(data[key], (int, float)) or data[key] <= 0:
                raise ValueError(f"Invalid value for {key}: {data[key]} (must be a positive number)")

        # Ensure log-transformable fields are valid
        data['Total_Area'] = float(data['Total_Area'])
        data['Price_per_SQFT'] = float(data['Price_per_SQFT'])
        data['Baths'] = float(data['Baths'])

        return data
    except Exception as e:
        raise ValueError(f"Input validation error: {e}")


# Function to predict cost using trained model
def predict_cost(attributes, scaler, W1, b1, W2, b2):
    """
    Predicts the cost based on input attributes using the trained neural network.
    """
    try:
        input_features = np.array([
            attributes['Balcony'],
            attributes['City_Bangalore'],
            attributes['City_Chennai'],
            attributes['City_Hyderabad'],
            attributes['City_Kolkata'],
            attributes['City_Mumbai'],
            attributes['City_Pune'],
            attributes['City_Thane'],
            np.log(attributes['Total_Area']),
            np.log(attributes['Price_per_SQFT']),
            np.log(attributes['Baths'])
        ]).reshape(1, -1)

        # Scale the input using the scaler
        scaled_input = scaler.transform(input_features)

        # Forward pass through the model
        Z1 = np.dot(scaled_input, W1) + b1
        A1 = np.maximum(0, Z1)  # ReLU activation
        Z2 = np.dot(A1, W2) + b2

        # Transform prediction back to the original cost scale (inverse of log transformation)
        predicted_log_cost = Z2.flatten()[0]
        predicted_cost = np.exp(predicted_log_cost)  # Inverse of log transformation
        return predicted_cost
    except Exception as e:
        print(f"Error in predicting cost: {e}")
        return None

# Function to match closest location in dataset
def match_closest_location(attributes, df, scaler, input_features):
    """
    Matches the given attributes to the most approximate location in the DataFrame.
    """
    try:
        input_array = np.array([
            attributes['Balcony'],
            attributes['City_Bangalore'],
            attributes['City_Chennai'],
            attributes['City_Hyderabad'],
            attributes['City_Kolkata'],
            attributes['City_Mumbai'],
            attributes['City_Pune'],
            attributes['City_Thane'],
            np.log(attributes['Total_Area']),
            np.log(attributes['Price_per_SQFT']),
            np.log(attributes['Baths'])
        ]).reshape(1, -1)

        scaled_input = scaler.transform(input_array)
        df_features = df[input_features].values
        scaled_df_features = scaler.transform(df_features)

        # Calculate Euclidean distances
        distances = np.linalg.norm(scaled_df_features - scaled_input, axis=1)
        closest_index = np.argmin(distances)

        # Retrieve closest match
        closest_match = df.iloc[closest_index]
        return closest_match
    except Exception as e:
        print(f"Error in finding closest match: {e}")
        return None

# Initialize Flask App
app = Flask(__name__)

CORS(app)

# Load the model and data
model_data = load_or_train_model()

if model_data:
    W1 = model_data['W1']
    b1 = model_data['b1']
    W2 = model_data['W2']
    b2 = model_data['b2']
    scaler = model_data['scaler']
    df = model_data['df']
    input_features = model_data['input_features']
else:
    print("Failed to load the model. Exiting app.")
    exit(1)

# Define API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Preprocess and validate input data
        try:
            validated_data = validate_and_preprocess_data(data)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        # Predict cost
        predicted_cost = predict_cost(validated_data, scaler, W1, b1, W2, b2)
        if predicted_cost is None:
            return jsonify({'error': 'Failed to predict cost'}), 500

        # Find closest match
        closest_match = match_closest_location(validated_data, df, scaler, input_features)
        if closest_match is None:
            return jsonify({'error': 'Failed to find closest match'}), 500

        # Prepare response
        response = {
            'predicted_cost': predicted_cost,
            'closest_match': {
                'City': closest_match['Location'],
                'Total Area': np.exp(closest_match['log_Total_Area']),
                'Cost': np.exp(closest_match['log_Cost']),
                'Price per SQFT': np.exp(closest_match['log_Price_per_SQFT']),
            }
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
