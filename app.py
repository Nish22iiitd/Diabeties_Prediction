from flask import Flask,request,app,render_template,Response
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

scaler=pickle.load(open('Model/std_scaler.pkl','rb'))
model=pickle.load(open('Model/model_pred2.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/diabeties_pred',methods=['GET','POST'])
def predict_datapoint():
    res=""
    if request.method=='POST':
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict_d=model.predict(data)
        if predict_d[0]==1: res="Diabetic"
        else: res="Non-Diabetic"
        return render_template('single_prediction.html',result=res)
    else: return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
