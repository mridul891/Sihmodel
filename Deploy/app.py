from flask import Flask, jsonify
import pickle
import os

app = Flask(__name__)

# Path to the pickle file
# Make sure this file exists in your directory or specify the correct path
PICKLE_FILE_PATH = 'model.pkl'


def load_pickle_data():
    """
    Loads data from a pickle file.
    """
    if not os.path.exists(PICKLE_FILE_PATH):
        return {"error": "Pickle file not found."}
    with open(PICKLE_FILE_PATH, 'rb') as f:
        data = pickle.load(f)
    return data


@app.route('/')
def home():
    return "Welcome to the Flask API!"


@app.route('/data', methods=['GET'])
def get_data():
    """
    Loads pickle data and returns it as JSON.
    """
    data = load_pickle_data()
    print(data)
    # Check if data was successfully loaded
    return data, 200


if __name__ == "__main__":
    app.run(debug=True)
