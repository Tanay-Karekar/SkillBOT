import pandas as pd
import numpy as np
import json
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

        # Check if the intent is "Introvert"
        if intent_name == "Introvert":
            # Load the encoder and model from files
            encoder_path = 'encodertest1.pkl'
            model_path = 'modeltest1.pkl'

            saved_encoder = load(encoder_path)
            saved_model = load(model_path)

            # Check if the user response is empty
            if not user_response:
                return "No user response available"

            # Create a DataFrame from the user responses
            user_responses_df = pd.DataFrame(user_responses)

            # Encode the user responses using the saved encoder
            user_responses_encoded = saved_encoder.transform(user_responses_df[categorical_features]).toarray()

            # Get the feature names after encoding
            encoded_feature_names = saved_encoder.get_feature_names_out(categorical_features)

            # Create a DataFrame with the encoded categorical features
            user_responses_encoded_df = pd.DataFrame(user_responses_encoded, columns=encoded_feature_names)

            # Create a DataFrame with the numerical features
            numerical_features_df = pd.DataFrame(user_responses_df[numerical_features], columns=numerical_features)

            # Combine numerical and encoded categorical features
            X = pd.concat([user_responses_encoded_df, numerical_features_df], axis=1)

            # Make job predictions for the user responses
            job_prediction = saved_model.predict(X)[0]

            # Store the job prediction in a response for the chatbot
            response = 'The Job Role That Best Suits Your Skills is "' + job_prediction + '"'
            
            # Save the user responses to a JSON file
            with open('user_responses.json', 'w') as json_file:
                json.dump(user_responses, json_file)

            return {
                'fulfillmentText': response
            }

    # Save the user responses to a JSON file
    with open('user_responses.json', 'w') as json_file:
        json.dump(user_responses, json_file)

    return "OK"

@app.route('/job_predictions')
def predict_job():
    # Load the new data
    new_data_path = 'user_responses.json'
    with open(new_data_path, 'r') as json_file:
        new_data = json.load(json_file)

    if not new_data:
        return "No user responses available"

    # Load the encoder and model from files
    encoder_path = 'encodertest1.pkl'
    model_path = 'modeltest1.pkl'

    saved_encoder = load(encoder_path)
    saved_model = load(model_path)

    # Check if the "Introvert" feature exists in the new data
    if "Introvert" in new_data:
        # Check if the "Memory Capability" column exists in the new data
        if "Memory Capability" not in new_data:
            return "Required information 'Memory Capability' is missing."

        new_data_encoded = saved_encoder.transform(pd.DataFrame(new_data, index=[0])[categorical_features]).toarray()
    else:
        return "No user responses available"

    # Get the feature names after encoding
    new_encoded_feature_names = saved_encoder.get_feature_names_out(categorical_features)

    # Create a DataFrame with the encoded categorical features
    new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=new_encoded_feature_names)

    # Create a DataFrame with the numerical features
    new_data_numerical = pd.DataFrame(pd.DataFrame(new_data, index=[0])[numerical_features], columns=numerical_features)

    # Combine numerical and encoded categorical features
    X_new = pd.concat([new_data_encoded_df, new_data_numerical], axis=1)

    # Make job predictions for new data
    job_predictions = saved_model.predict(X_new)
    job_predictions = job_predictions.tolist()

    # Return the job predictions as a response
    return json.dumps(job_predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
