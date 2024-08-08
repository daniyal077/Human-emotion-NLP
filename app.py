from flask import Flask, request, render_template
import numpy as np
import nltk
import re
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('emotion_model.h5')
le = pickle.load(open('labelEncoder.pkl', 'rb'))
vocab_info = pickle.load(open('vocab_info.pkl', 'rb'))

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))



def emotion_decoder(emotion):
    message=[]
    if emotion==1:
        message.append('Fear')
    elif emotion==2:
        message.append('Joy')
    elif emotion==3:
        message.append('Love')
    elif emotion==4:
        message.append('Sadness')
    elif emotion==5:
        message.append('Suprise')
    else:
        message.append('Anger')
    
    return(message[0])


# Sentence cleaning function
def sentence_clean(sentence):
    text = re.sub('[^a-zA-Z]', ' ', sentence).lower()
    words = text.split()
    words = [word for word in words if word not in stopwords]
    cleaned_text = ' '.join(words)
    one_hot_word = one_hot(input_text=cleaned_text, n=11000)
    pad = pad_sequences(sequences=[one_hot_word], maxlen=300, padding='pre')
    return pad

# Prediction function
def paredict_result(sentence):
    sentence = sentence_clean(sentence)
    prediction = model.predict(sentence)
    result = np.argmax(prediction)
    result = emotion_decoder(result)
    prob = np.max(prediction)
    return result, prob

# Flask application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        if text:
            result, prob = paredict_result(text)
            print("Result:", result)  
            print("Probability:", prob)  
            return render_template('home.html', result=result, prob=prob)
        else:
            message = 'Please write text'
            return render_template('home.html', message=message)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
