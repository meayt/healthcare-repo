from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (make sure the model is saved as 'rf_model.pkl')
model = joblib.load('meet/rf_model.pkl')

# Load LabelEncoder (make sure to save and load the LabelEncoder in a similar way)
label_encoder = joblib.load('meet/label_encoder.pkl')

@app.route('/')
def home():
    return render_template('meet/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        bmi = float(request.form['bmi'])

        # Prepare the input for prediction
        input_data = np.array([[age, height, weight, bmi]])

        # Predict using the Random Forest model
        prediction = model.predict(input_data)

        # Decode the prediction (i.e., convert numeric label back to category)
        bmi_class = label_encoder.inverse_transform(prediction)

        # Return the result to the user
        return render_template('meet/index.html', prediction_text=f'The person is classified as: {bmi_class[0]}')


    except Exception as e:
        return render_template('meet/index.html', prediction_text="Error occurred, please check your inputs.")

if __name__ == "__main__":
    app.run(debug=True)
