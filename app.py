from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)
ridge=pickle.load(open('Models/ridge.pkl','rb'))
scaler=pickle.load(open('Models/scaler.pkl','rb'))
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata" ,methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        scaled_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result1=ridge.predict(scaled_data)

        print([Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region])
        print(scaled_data)
        print(result1[0][0])
     
        return render_template('home.html',result=result1[0][0])
        
    else:
        return render_template('home.html')


if __name__=='__main__':
    app.run(host='0.0.0.0')