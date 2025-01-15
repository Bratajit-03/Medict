from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import os



app = Flask(__name__)

#lung cancer prediction
def lpredict(values, dic):
    if len(values) == 15:
        model = pickle.load(open('models/lung_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

#stroke prediction
def strokepredict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/stroke_prediction (1).pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

#heart-disease prediction
def heart_diseasepredict(values, dic):
    if len(values) == 13:
        model = pickle.load(open('models/heart_disease.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    
#CKD prediction
def ckdpredict(values, dic):
    if len(values) == 21:
        model = pickle.load(open('models/ckd.pkl','rb'))
        values = np.asarray(values)
        print("Prediction:", model.predict(values.reshape(1, -1))[0])
        return model.predict(values.reshape(1, -1))[0]

#Liver prediction
def liverpredict(values, dic):
    if len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        print("Prediction:", model.predict(values.reshape(1, -1))[0])
        return model.predict(values.reshape(1, -1))[0]

#Parkinson's prediction
def parkinsonpredict(values, dic):
    if len(values) == 19:
        model = pickle.load(open('models/parkinson.pkl','rb'))
        values = np.asarray(values)
        print("Prediction:", model.predict(values.reshape(1, -1))[0])
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/disease_index")
def disease_index():
    return render_template('Disease_Index.html')

@app.route("/detect")
def predictorbd():
    return render_template('predictorbd.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart_disease.html')

@app.route("/stroke", methods=['GET', 'POST'])
def strokePage():
    return render_template('stroke2.0.html')

@app.route("/lungCancer", methods=['GET', 'POST'])
def lungCancerPage():
    return render_template('lung_cancer.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/heartdisease", methods=['GET', 'POST'])
def HeartDiseasePage():
    return render_template('heart_disease.html')

@app.route("/ckd", methods=['GET', 'POST'])
def CKDPage():
    return render_template('ckd.html')

@app.route("/parkinson", methods=['GET', 'POST'])
def parkinsonPage():
    return render_template('parkinson.html')



@app.route("/lungcancerpredict", methods=['POST', 'GET'])
def lungcancerpredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = lpredict(to_predict_list, to_predict_dict)
    except:
        message = "Please Enter Correct Data"
        return render_template("home.html", message=message)

    return render_template('lungcancerpredict.html', pred=pred)

@app.route("/strokepredict", methods=['POST', 'GET'])
def strokepredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = strokepredict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter Correct Data."
        return render_template("home.html", message=message)

    return render_template('strokepredict.html', pred=pred)

@app.route("/heartpredict", methods=['POST', 'GET'])
def heart_diseasepredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = heart_diseasepredict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter Correct Data."
        return render_template("home.html", message=message)

    return render_template('heart_diseasepredict.html', pred=pred)

@app.route("/ckdpredict", methods=['POST', 'GET'])
def ckdpredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = ckdpredict(to_predict_list, to_predict_dict)
            print(pred)
    except:
        message = "Please enter Correct Data."
        return render_template("home.html", message=message)

    return render_template('ckdpredict.html', pred=pred)

@app.route("/liverpredict", methods=['POST', 'GET'])
def liverpredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = liverpredict(to_predict_list, to_predict_dict)
            print(pred)
    except:
        message = "Please enter Correct Data."
        return render_template("home.html", message=message)

    return render_template('liverpredict.html', pred=pred)

@app.route("/parkinsonpredict", methods=['POST', 'GET'])
def parkinsonpredictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = parkinsonpredict(to_predict_list, to_predict_dict)
            print(pred)
    except:
        message = "Please enter Correct Data."
        return render_template("home.html", message=message)

    return render_template('parkinsonpredict.html', pred=pred)



if __name__ == '__main__':
	app.run(debug = True)