from flask import Flask, request, render_template, redirect, url_for
import numpy as np 
import pandas as pd
import joblib

app = Flask(__name__)
prediction = joblib.load("winePredictionModel.sav")

@app.route('/predict', methods=['GET', 'POST'])
def predictWine():
    if request.method == "POST":
        details = {
        'fixed acidity' : [float(request.form["fa"])],
        'volatile acidity' : [float(request.form["va"])],
        'citric acid' :  [float(request.form["ca"])],
        'residual sugar':  [float(request.form["rs"])],
        'chlorides' : [float(request.form["chl"])],
        'free sulfur dioxide' :[float(request.form["fsd"])],
        'total sulfur dioxide' :[float(request.form["tsd"])],
        'density': [float(request.form["den"])],
        'pH': [float(request.form["ph"])],
        'sulphates' : [float(request.form["sul"])],
        'alcohol': [float(request.form["alc"])]
        }
        df = pd.DataFrame(data=details)
        quality = ['Good', 'Medium', 'Bad']
        ans = prediction.predict(df)
        return "Quality is " + str(quality[ans[0]]) 
    return render_template('predictionForm.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
    