from flask import Flask,render_template,url_for,request,jsonify
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
    return render_template('index.html')

@app.route('/background_process')
def background_process():
	try:
		lang = request.args.get('review',type=str)
		if lang.lower() == 'pos':
			return jsonify({"result": "Positive","toggle":1})
		elif lang.lower() == 'neg':
			return jsonify({"result": "Negative","toggle":0})
	except Exception as e:
		return str(e)


if __name__ == '__main__':
    app.run(debug=True)