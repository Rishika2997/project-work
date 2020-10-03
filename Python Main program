from flask import Flask, render_template, request,session,logging,flash,url_for,redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from flask_mail import Mail
import os
import secrets
# import pickle
import numpy as np
from sklearn.externals import joblib

import ast



import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#import sexmachine.detector as gender
import gender_guesser.detector as gender
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
#from sklearn.cross_validation import StratifiedKFold, train_test_split

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
# # import pickle

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

local_server = True
app = Flask(__name__,template_folder='template')
app.secret_key = 'super-secret-key'

app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = params['gmail_user']
app.config['MAIL_PASSWORD'] = params['gmail_password']
mail = Mail(app)

if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)

class Contact(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(80), nullable=False)
    lname = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(12), nullable=False)
    message = db.Column(db.String(50), nullable=False)
    date = db.Column(db.String(12), nullable=True)

class Register(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email=db.Column(db.String(80), nullable=False)
    password = db.Column(db.String(12), nullable=False)
    password2 = db.Column(db.String(120), nullable=False)
    bday=db.Column(db.String(120), nullable=False)
    gender=db.Column(db.String(120), nullable=False)
   

class Forgetpassword(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    email=db.Column(db.String, nullable=False)
    token=db.Column(db.String(50), nullable=False)





@app.route("/")
def home():
    return render_template('index.html',params=params)

@app.route("/termofservice")
def termofservice():
    return render_template('termofservice.html',params=params)

@app.route("/facebookdashboard")
def facebookdashboard():
    return render_template('facebookdashboard.html',params=params)


@app.route("/detect", methods=['GET','POST'])
def detect():
    if(request.method=="POST"):
        name=request.form['name']
        name=request.form['lang_code']
        statuses_count=float(request.form['statuses_count'])
        followers_count=float(request.form['followers_count'])
        friends_count=float(request.form['friends_count'])
        favourites_count=float(request.form['favourites_count'])
        listed_count=float(request.form['listed_count'])
        pred_args=[statuses_count,followers_count,friends_count,favourites_count, listed_count]
        pred_args_arr=np.array(pred_args)
        pred_args_arr=pred_args_arr.reshape(1,-1)
        # import pdb;pdb.set_trace()
        print(pred_args_arr)
        pickle_model = joblib.load('filename1.pkl')
        print(pickle_model)

        # 
        # # from fakesvmprofilepratice import read_datasets, extract_features,train
        # print(model)
        # # import thread
        model_prediction=pickle_model.predict(pred_args_arr)
        print(model_prediction)
        model_predict=round(float(model_prediction))
        print(model_predict)
        return render_template("svm.html", params=params, prediction_text=model_predict)
    else:
        return render_template("svm.html",params=params)


    # model=pickle.load(open('model.pkl', 'rb'))
    
    # int_features=[int(x) for x in request.form.values()]
   
    # final_features = [np.array(int_features)]

    # prediction=model.predict(final_features)
    # output=round(prediction[0])

    # return render_template("svm.html", prediction_text='creditcard fraud detect be $ {}'.format(output),params=params)

@app.route("/about")
def about():
    return render_template('about.html',params=params)


@app.route("/login", methods=['GET','POST'])
def login():
    if('email' in session and session['email']):
        return render_template('dashboard.html',params=params)

    if (request.method== "POST"):
        email = request.form["email"]
        password = request.form["password"]
        
        login = Register.query.filter_by(email=email, password=password).first()
        if login is not None:
            session['email']=email
            return render_template('dashboard.html',params=params)
        else:
            flash("plz enter right password")
    return render_template('login.html',params=params)


@app.route("/register", methods=['GET','POST'])
def register():
    if(request.method=='POST'):
        name = request.form.get('name')
        email=request.form.get('email')
        password = request.form.get('password')
        password2 = request.form.get('password2')
        bday=request.form.get('bday')
        gender=request.form.get('gender')
      

        if (password==password2):
            entry = Register(name=name,email=email,password=password,password2=password2, bday=bday, gender=gender)
            db.session.add(entry)
            db.session.commit()
            return redirect(url_for('register'))
        else:
            flash("plz enter right password")
    return render_template('register.html',params=params)

@app.route("/contact", methods = ['GET', 'POST'])
def contact():
    if(request.method=='POST'):
        '''Add entry to the database'''
        fname = request.form.get('fname')
        lname = request.form.get('lname')
        email = request.form.get('email')
        message = request.form.get('message')
        entry = Contact(fname=fname, lname = lname, email = email, message = message,date= datetime.now() )
        db.session.add(entry)
        db.session.commit()
        mail.send_message('New message from ' + fname,
                          sender=email,
                          recipients = [params['gmail_user']],
                          body = message + "\n" + message
                          )
    return render_template('contact.html',params=params)


@app.route("/logout", methods = ['GET','POST'])
def logout():
    session.pop('email')
    return redirect(url_for('login'))


@app.route("/forgetpassword", methods = ['GET','POST'])
def forgetpassword():
    if(request.method=='POST'):
        email=request.form.get('email')
        token=secrets.token_urlsafe(6)
        send_reset_email(email,token)
        entry=Forgetpassword(email=email,token=token)
        db.session.add(entry)
        db.session.commit()
        return render_template('forgetpassword.html',params=params)
    else:
        return render_template("forgetpassword.html", params=params)

def send_reset_email(email,token):
    message_body="http://127.0.0.1:5000/reset_password/"+token
    mail.send_message('New message from ' ,
                          sender="params['gmail_user']",
                          recipients = [email],
                          body = message_body 
                          )

@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_password(token):
    if(request.method=='POST'):
        email=request.form.get('email')
        password=request.form.get('password')
        '''update query'''
        record=Register.query.filter_by(email=email).first()
      
        '''
        import pdb;pdb.set_trace()
        '''
        if record:
            record.password=password
            record.password2=password
            db.session.add(record)
            db.session.commit()
            return redirect(url_for('login'))
        else:
            pass

    else:
        record = Forgetpassword.query.filter_by(token=token).first()
        if record:
                 return render_template('resetpassword.html', email=record.email)
                 

if __name__ == "__main__":
    
    app.run(debug=True)
