import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("baseline_personality.pkl")
print('Model has been loaded for inference....')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    gender = request.form['gender']
    if gender == 'Male':
        gender = 0
    else:
        gender = 1
    age =  int(request.form['age'])
    openness =  int(request.form['open'])
    agitate =  int(request.form['agitate'])
    discipline =  int(request.form['discipline'])
    agree =  int(request.form['agree'])
    social =  int(request.form['social'])

    if age < 1 or age > 100:
        return render_template('index.html', prediction_text = 'Please enter valid age')
    
    features = np.array([gender, age, openness, agitate, discipline, agree, social])

    output = model.predict([features])[0]

    print(output)

    return render_template('index.html', prediction_text = f'Your personality is {output}')


if __name__ == "__main__":
    app.run(debug = True)
    #app.run(host='0.0.0.0', port=8080)
    