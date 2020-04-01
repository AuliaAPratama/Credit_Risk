from flask import Flask, render_template, request
import joblib
import plotly
import json
import plotly.graph_objs as go
import chart_studio.plotly as csp
from sklearn.preprocessing import binarize
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('input.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        body = request.form
        umur = float(body['DAYS_AGE'])*365
        cicilan = float(body['ANNUITY'])*10
        kerja = float(body['DAYS_WORK'])*365
        pendapatan = float(body['INCOME'])*10 
        pilihan = body['INCOME_TYPE']
        combine = [umur,cicilan,kerja,pendapatan] + pilihanku[f'{pilihan}']
        prediksi = model.predict([combine])[0]
        proba = model.predict_proba([combine])[0]
        plot = go.Pie(labels=['Late Payment','Ontime Payment'],values=proba)
        graphJson = json.dumps([plot],cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('predict.html', data=body, pred=prediksi, tebak_plot=graphJson)


if __name__ == '__main__':
    pilihanku = {
        'Commercial associate' : [1,0,0,0,0],
        'Pensioner' : [0,1,0,0,0],
        'State servant' : [0,0,1,0,0],
        'Unemployed' : [0,0,0,1,0],
        'Working' : [0,0,0,0,1]
    }
    model = joblib.load('modelJoblib')
    app.run(debug=True)