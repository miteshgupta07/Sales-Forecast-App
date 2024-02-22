# Import necessary libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a Flask web application instance
app = Flask(__name__)

# Load the pre-trained machine learning model using Pickle
model = pickle.load(open('model.pkl', 'rb'))

# Define the route for the home page, which renders the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions based on form data
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the input features from the form data and convert to integers
    int_features = [int(float(x)) for x in request.form.values()]

    # Create a NumPy array from the input features
    final_features = [np.array(int_features)]

    # Make a prediction using the loaded machine learning model
    prediction = model.predict(final_features)

    # Round the prediction to two decimal places
    output = round(prediction[0], 2)

    # Render the index.html template with the prediction result
    return render_template('index.html', prediction_text='Forecasted Sales: $ {}'.format(output))

# Define the route for making predictions based on JSON data
@app.route('/results', methods=['POST'])
def results():
    # Get the JSON data sent with the request
    data = request.get_json(force=True)

    # Make a prediction using the loaded machine learning model
    prediction = model.predict([np.array(list(data.values()))])

    # Get the output prediction
    output = prediction[0]

    # Return the prediction as a JSON response
    return jsonify(output)

# Run the Flask web application in debug mode
if __name__ == "__main__":
    app.run(debug=True)
