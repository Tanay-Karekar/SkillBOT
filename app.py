import pandas as pd
import numpy as np
import json
import pickle
from flask import Flask, render_template, request, make_response
from joblib import load

app = Flask(__name__)

# Create a dictionary to store user responses
user_responses = {}

# Define the list of intent names
intent_names = [
    "Logical quotient rating", "Hackathons", "Coding Skills Rating", "Public Speaking Points",
    "self-learning capability?", "Extra-courses did", "certifications", "Workshop",
    "Reading and Writing Skills", "Memory Capability", "Interested Subjects",
    "Interested Career Area", "Type of company want to settle in?", "Taken inputs from seniors or elders",
    "Interested Type of Books", "Management or Technical", "Hard/Smart worker", "worked in teams ever?", "Introvert"
]

# Define numerical and categorical features
numerical_features = ['Logical quotient rating', 'Hackathons', 'Coding Skills Rating', 'Public Speaking Points']
categorical_features = ['self-learning capability?', 'Extra-courses did', 'certifications', 'Workshop',
                        'Reading and Writing Skills', 'Memory Capability', 'Interested Subjects',
                        'Interested Career Area', 'Type of company want to settle in?',
                        'Taken inputs from seniors or elders', 'Interested Type of Books',
                        'Management or Technical', 'Hard/Smart worker', 'worked in teams ever?', 'Introvert']

@app.route('/')
def hello_world():
    # Clear user responses when the page is reloaded
    user_responses.clear()

    # Create an empty JSON file
    with open('user_responses.json', 'w') as json_file:
        json.dump(user_responses, json_file)
    return render_template('frontend.html')

@app.route('/products')
def products():
    return 'This is products'

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    # Extract the intent name from the data
    intent_name = data.get('queryResult').get('intent').get('displayName')
    print("Intent Name:", intent_name)  # Debug statement

    # Extract the user response for the intent
    user_response = data.get('queryResult').get('queryText')
    print("User Response:", user_response)  # Debug statement

    # Store the intent name and user response in the dictionary
    if intent_name and user_response and intent_name in intent_names:
        print(f"Entered '{intent_name}' condition")  # Debug statement
        if intent_name not in user_responses:
            user_responses[intent_name] = []
        user_responses[intent_name].append(user_response)

    # Save the user responses to a JSON file
    with open('user_responses.json', 'w') as json_file:
        json.dump(user_responses, json_file)

    # Check if the intent is "Introvert" and perform the prediction logic
    if intent_name == "Introvert":
        # Load the encoder and model from files
        encoder_path = 'encodertest1.pkl'
        model_path = 'modeltest1.pkl'

        with open(encoder_path, 'rb') as f:
            saved_encoder = pickle.load(f)

        with open(model_path, 'rb') as f:
            saved_model = pickle.load(f)

        # Load the new data
        new_data_path = 'user_responses.json'
        with open(new_data_path, 'r') as json_file:
            new_data = json.load(json_file)

        if not new_data:
            return "No user responses available"

        # Encode the new data using the saved encoder
        new_data_encoded = saved_encoder.transform(pd.DataFrame(new_data, index=[0])[categorical_features]).toarray()

        # Get the feature names after encoding
        new_encoded_feature_names = saved_encoder.get_feature_names_out(categorical_features)

        # Create a DataFrame with the encoded categorical features
        new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=new_encoded_feature_names)

        # Combine numerical and encoded categorical features
        new_data_numerical = pd.DataFrame(new_data, index=[0])[numerical_features]
        X_new = pd.concat([new_data_numerical, new_data_encoded_df], axis=1)

        # Reorder the columns to match the feature names used during training
        X_new = pd.concat([X_new[numerical_features], X_new[new_encoded_feature_names]], axis=1)

        # Make job predictions for new data
        job_predictions = saved_model.predict(X_new)
        print("The Job Role That Best Suits Your Skills is:", job_predictions)

        # Store the job prediction in a response for the chatbot
        response = 'The Job Role That Best Suits Your Skills is "' + job_predictions[0] + '"'

        return {
            'fulfillmentText': response
        }

    return "OK"

@app.route('/job_predictions')
def predict_job():
    # Load the new data
    new_data_path = 'user_responses.json'
    with open(new_data_path, 'r') as json_file:
        new_data = json.load(json_file)

    if not new_data:
        return "No user responses available"

    encoder_path = 'encodertest1.pkl'
    model_path = 'modeltest1.pkl'

    saved_encoder = load(encoder_path)
    saved_model = load(model_path)

    # Check if the "Introvert" feature exists in the new data
    if "Introvert" in new_data:
        if "Memory Capability" not in new_data:
            return "Required information 'Memory Capability' is missing."

        new_data_encoded = saved_encoder.transform(pd.DataFrame(new_data, index=[0])[categorical_features]).toarray()
    else:
        return "No user responses available"

    # Get the feature names after encoding
    new_encoded_feature_names = saved_encoder.get_feature_names_out(categorical_features)
   # Create a DataFrame with the encoded categorical features
    new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=new_encoded_feature_names)

    # Create a new DataFrame with the correct feature names and order
    new_data_numerical = pd.DataFrame(new_data, index=[0])[numerical_features]
    X_new = pd.concat([new_data_numerical, new_data_encoded_df], axis=1)

    # Reorder the columns to match the feature names used during training
    feature_names_used = numerical_features + new_encoded_feature_names
    X_new = X_new[feature_names_used]

    # Convert categorical features to integers
    categorical_cols = [
        col for col in X_new.columns if col.startswith('self-learning capability?')
        or col.startswith('Extra-courses did') or col.startswith('certifications')
        or col.startswith('Workshop') or col.startswith('Reading and Writing Skills')
        or col.startswith('Memory Capability') or col.startswith('Interested Subjects')
        or col.startswith('Interested Career Area') or col.startswith('Type of company want to settle in?')
        or col.startswith('Taken inputs from seniors or elders') or col.startswith('Interested Type of Books')
        or col.startswith('Management or Technical') or col.startswith('Hard/Smart worker')
        or col.startswith('worked in teams ever?') or col.startswith('Introvert')
    ]

    X_new[categorical_cols] = X_new[categorical_cols].astype(int)

    # Make job predictions for new data
    job_predictions = saved_model.predict(X_new)
    job_predictions = job_predictions.tolist()

    # Return the job predictions as a response
    return json.dumps(job_predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)