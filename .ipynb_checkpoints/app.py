from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        data = request.form.to_dict()
        # Convert data to a dataframe
        df = pd.DataFrame([data])
        # Convert all values to numeric
        df = df.apply(pd.to_numeric)
        
        # Predict using the model
        prediction = model.predict(df)
        
        # Return the result
        result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
        return render_template('result.html', prediction=result)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
