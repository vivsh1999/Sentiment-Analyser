from flask import Flask,render_template,url_for,request
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC, LinearSVC
import pickle

app = Flask(__name__)


    

@app.route('/',methods=['GET','POST'])
def home():
    prediction=-1
    output=""
    return render_template('index.html',prediction=prediction,output=output)


if __name__ == '__main__':
    app.run(debug=True)