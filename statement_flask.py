#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from flask import Flask, render_template, request
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

model = pickle.load(open("saved_model.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text_value = [request.form.values()]
    seq = tokenizer.texts_to_sequences(text_value)
    padded = pad_sequences(seq, maxlen=150)
    pred = model.predict(padded)
    
    classes = ['Politics', 'Love', 'Heavy Emotion', 'Health', 'Animals',
       'Science', 'Joke', 'Compliment', 'Religion', 'Self', 'Education']
    
    pred_class = classes[np.argmax(pred)]
                         
    return render_template('index.html', prediction_text='Context of the statement is :{}'.format(pred_class))
if __name__ == '__main__':
    app.run()


# In[ ]:




