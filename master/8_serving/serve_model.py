"""
Flask API to serve the model
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

COLNAMES = [
    'surface_reelle_bati',
    'nombre_pieces_principales',
    'code_postal',
    'total_surface_lots'
    ]

# Load the model
model = joblib.load('pipeline.joblib')

# Create the Flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def help_endpoint():
    # Provide information about available endpoints
    endpoints_info = [
        {
            'url': '/',
            'description': 'Provides help and information about available endpoints.'
        },
        {
            'url': '/predict',
            'description': 'Returns predicted house prices based on input data.',
            'parameters': [
                {
                    'name': 'surface',
                    'description': 'Surface of the house in square meters.',
                    'type': 'float'
                },
                {
                    'name': 'rooms',
                    'description': 'Number of rooms in the house.',
                    'type': 'float'
                },
                {
                    'name': 'terrain',
                    'description': 'Total surface of the lots in square meters.',
                    'type': 'float'
                },
                {
                    'name': 'zip',
                    'description': 'Zip code of the house.',
                    'type': 'int'
                }
            ],
            'example': '/predict?surface=50&rooms=3&terrain=0&zip=75001'
        }
    ]
    return jsonify({'endpoints': endpoints_info})


@app.route('/predict', methods=['GET'])
def predict():
    """
    Predict the output of the model
    """

    # Get the parameters from the request
    args = request.args

    try:
        # Extract the parameters
        surface = float(args.get('surface'))
        rooms = float(args.get('rooms'))
        terrain = float(args.get('terrain'))
        zip = int(args.get('zip'))

        # Convert to dataframe
        data = [[surface, rooms, zip, terrain]]
        data = pd.DataFrame(
            data, columns=COLNAMES)

        print(data)

        # Predict the output
        output = model.predict(data)

    except Exception as e:
        return jsonify({'error': str(e)})

    response = jsonify({'predicted_price': str(output[0])})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
