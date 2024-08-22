from flask import Flask,jsonify,render_template,request
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

application=Flask(__name__)

ridge_model=pickle.load(open('ridge.pkl','rb'))
standard_scaler=pickle.load(open('scaler.pkl','rb'))

@application.route("/")
def index():
    return render_template('index.html')

@application.route("/predict_data", methods=['GET','POST'] )
def predict_datapoint(): 
    if request.method=="POST":
        temperature=float(request.form.get('temperature'))
        RH=float(request.form.get('RH'))
        WS=float(request.form.get('WS'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        region=float(request.form.get('region'))

        new_data=standard_scaler.transform([[temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,region]])
        result=ridge_model.predict(new_data)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    application.run(debug=True)


