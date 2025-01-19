from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = 'model.pkl'
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it is in the same directory as this script.")
    model = pickle.load(open(MODEL_PATH, 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

@app.route('/')
def index():
    """
    Render the home page (index.html).
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle crop recommendation form submission and display results.
    """
    try:
        # Ensure the model is loaded
        if model is None:
            return "Model not loaded. Please check the server logs for more details."

        # Retrieve form data
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        print(f"Received input: {nitrogen}, {phosphorus}, {potassium}, {temperature}, {humidity}, {ph}, {rainfall}")

        # Prepare input for prediction
        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Make the prediction
        prediction = model.predict(input_features)[0]

        # Render the result page with the prediction
        return render_template('result.html', crop=prediction)
    
    except Exception as e:
        # Catching any errors during prediction
        print(f"Error during prediction: {e}")
        return f"An error occurred during prediction: {e}"

if __name__ == '__main__':
    # Run the Flask app in debug mode to see detailed errors
    app.run(debug=True)
