from flask import Flask,render_template,url_for,request,jsonify
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC, LinearSVC
import pickle
from bs4 import BeautifulSoup      #for html removal       
import re                        #for number removal
import nltk                      #for stopwords removal
import nltk.data


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.stem import PorterStemmer


vec_file=open("data/vectorizer.pickle","rb")
linsvc_file=open("data/linear_svc.pickle","rb")
tfidf_file=open("data/tfidf_transformer.pickle","rb")
vectorizer=pickle.load(vec_file)
linear_svc=pickle.load(linsvc_file)
tfidf_transformer=pickle.load(tfidf_file)




def rev2words(rev):
    #making beautyful/removing markups and tags
    words=BeautifulSoup(rev,features="html.parser").get_text()
    #removing digits,symbols,and everything except plain letters
    letters = re.sub("[^a-zA-Z0-9!?'-]", " ", words)
    lemma=WordNetLemmatizer()
    words_arr=[lemma.lemmatize(w) for w in word_tokenize(str(letters).lower())]
    
    stop_words=set(stopwords.words("english"))
    meaningful_words=[w for w in words_arr if w not in stop_words]
    return " ".join(meaningful_words)



app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/background_process')
def background_process():
	try:
		lang = request.args.get('review',type=str)
		rev=lang.lower()
		#making beautyful/removing markups and tags
		words=BeautifulSoup(rev,features="html.parser").get_text()
		#removing digits,symbols,and everything except plain letters
		letters = re.sub("[^a-zA-Z0-9!?'-]", " ", words)
		lemma=WordNetLemmatizer()
		words_arr=[lemma.lemmatize(w) for w in word_tokenize(str(letters).lower())]
		
		stop_words=set(stopwords.words("english"))
		meaningful_words=[w for w in words_arr if w not in stop_words]
		
		clean_rev=[" ".join(meaningful_words)]
		test_data_features = vectorizer.transform(clean_rev)            #Vectorize Test Data
		test_data_features = test_data_features.toarray()
		test_tfidf=tfidf_transformer.transform(test_data_features)
		pred=linear_svc.predict(test_tfidf)
		pred=pred[0]
		if(pred==1):
			text="Positive"
		else:
			text="Negative"
		return jsonify({"result": text,"toggle":int(pred)})
		
			
	except Exception as e:
		return str(e)


if __name__ == '__main__':
    app.run(debug=True)



