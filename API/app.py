from flask import Flask, render_template, jsonify,request
import joblib
import os
import numpy as np

app = Flask(__name__)

print(os.getcwd())
os.chdir('/Users/ketkiambekar/Documents/GitHub-ketkiambekar/spam-classifier/API')
my_model= joblib.load("model/spam_classifier.joblib")
dictionary= joblib.load("model/dictionary.joblib")

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/response', methods=['POST'])
def response():
    msg = request.form.get("msg")
    messages=[]
    messages.append(msg)
    msg_matrix = transform_text(messages, dictionary)
    verdict = predict_from_naive_bayes_model(my_model, msg_matrix)
    return render_template("index.html", origmsg=msg, verdict=verdict)

def transform_text(messages, word_dictionary):
    vocab =  sorted(word_dictionary.keys())
    vocab_length = len(word_dictionary.keys())
    num_messages = len(messages)
    arr = np.zeros((num_messages,vocab_length), dtype=np.float128)
    for i in range(0,num_messages):
        words=get_words(messages[i])
          
        for word in words:
            if word in vocab:
                j = vocab.index(word)
                arr[i][j]+=1
    return arr

def get_words(message):
    #remove punctuations in the message
    punc='~`!@#$%^&*(),.:\";\'-+=_0123456789?£'
    for p in punc:
        message = message.replace(p,'')
    words =  message.lower().split(' ')
    return words

def predict_from_naive_bayes_model(model, matrix):
    phi_y0, phi_y1, phi_y = model
    matrix[matrix>1]=1
    y1 = phi_y1*matrix
    y1[y1==0]=1 
    y0 = phi_y0*matrix
    y0[y0==0]=1 
    num_messages, vocab = matrix.shape
    p1= np.exp(np.sum(np.log(y1), axis=1))*phi_y
    #denominator = numerator + np.exp(np.sum(np.log(y0), axis=1))*(1-phi_y)
    p0= np.exp(np.sum(np.log(y0), axis=1))*(1-phi_y)
   
    for i in range(0,len(p0)):
        if p0[i]>=p1[i]:
            return "Ham"
        else:
            return "Spam"

if __name__ == "__main__":
    app.run()