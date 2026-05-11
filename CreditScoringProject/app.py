from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    income = float(request.form['income'])
    debt = float(request.form['debt'])
    payment_history = int(request.form['payment_history'])
    loan_amount = float(request.form['loan_amount'])

    features = np.array([[income, debt, payment_history, loan_amount]])

    prediction = model.predict(features)

    result = ""

    if prediction[0] == 1:
        result = "Customer is Creditworthy"
    else:
        result = "Customer is NOT Creditworthy"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)