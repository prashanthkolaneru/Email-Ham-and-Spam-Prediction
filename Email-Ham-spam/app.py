#step -1 # Importing flask module in the project is mandatory 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vt = TfidfVectorizer(lowercase=False,vocabulary=None,tokenizer=None)

def text_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    

    return " ".join(text)
#Step -2 Flask constructor takes the name of  
# current module (__name__) as argument.app = Flask(__name__)

app = Flask(__name__)

#Step -3 Load Trained  Model
model = pickle.load(open('logistic.pkl', 'rb'))
one  = pickle.load(open('logistic_1.pkl', 'rb'))

# Step -4 The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #Data = (float(x) for x in request.form.values())
    message = request.form['message']
		
    
    data = {'Text':[message]}
    
   
    
    df1 = pd.DataFrame(data)
    print(df1)
    preprocessing = df1['Text'].copy()
    preprocessing = preprocessing.apply(text_process)
   
    x = one.transform(preprocessing)
    x = x.todense()
    output = model.predict(x)

    
    
    res = output
    return render_template('index.html', prediction_text=' The Email is {}'.format(res))


# main driver function
 # run() method of Flask class runs the application  
    # on the local development server.
if __name__ == "__main__":
    app.run(debug=True)

