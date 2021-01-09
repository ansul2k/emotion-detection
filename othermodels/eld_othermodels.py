"""
Created on Oct 19, 2020
CS594: Deep Learning for Natural Language Processing
University of Illinois at Chicago
Fall 2020
Project
"""

import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GRU, SimpleRNN
import pandas as pd
import string
import spacy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
spacy_nlp = spacy.load('en_core_web_sm')

#Load csv
train_df = pd.read_csv('dataset/train_sent_emo.csv',usecols=['Utterance','Emotion'])
dev_df = pd.read_csv('dataset/dev_sent_emo.csv',usecols=['Utterance','Emotion'])
test_df = pd.read_csv('dataset/test_sent_emo.csv',usecols=['Utterance','Emotion'])

print(train_df.shape)
print(dev_df.shape)
print(test_df.shape)

def load_glove():
    embeddings_dict= {}
    with open("Embeddings/glove.6B.300d.txt", 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

unigrams = {}

def countUnigrams(word):
    # finding unique words
    unigrams[word] = unigrams.get(word, 0)+1
    return unigrams

def get_tokens(doc):
    whiteSpaceToken = re.split("\\W+", doc)
    tokens = []
    # replacing a list of special characters by blank space
    transtable = str.maketrans('', '', string.punctuation)
    for word in whiteSpaceToken:
        tokens.append(word.lower().translate(transtable))
    return tokens

def removeStopWords(tokens):
    spacy_stopwords = spacy_nlp.Defaults.stop_words
    tokenList= []
    for word in tokens:
        if(word not in spacy_stopwords):
            tokenList.append(word)
    return tokenList

def cleanText(tokens):
    cleanText=[]
    for word in tokens:
        # cleaning text by keeping only alphabetical words & words keep with more than 2 letters
        if (word.isalpha() and len(word)>2):
            cleanText.append(word)
            countUnigrams(word)
    return cleanText

#Preprocessing
def preprocess(textDocs):     
    tokens = []
    for doc in textDocs:
        tokens.append(get_tokens(doc))
    
    tokensWOStopWords = []
    for doc in tokens:
        tokensWOStopWords.append(removeStopWords(doc))

    cleanWords = []
    for doc in tokensWOStopWords:
        cleanWords.append(cleanText(doc))
    
    return cleanWords

train_df['Emotion'].unique()
X_train = train_df['Utterance']
Y_train = train_df['Emotion']
X_test = test_df['Utterance']
Y_test = test_df['Emotion']
X_dev = dev_df['Utterance']
Y_dev = dev_df['Emotion']

Y_train = pd.get_dummies(Y_train)
Y_test = pd.get_dummies(Y_test)
Y_dev = pd.get_dummies(Y_dev)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train) 

multiplier = 1
# padding sequences to ensure all rows are of equal length
MAX_SEQUENCE_LENGTH =  10 

sequences = tokenizer.texts_to_sequences(X_train) 
X_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

sequences = tokenizer.texts_to_sequences(X_dev) 
X_dev = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

sequences = tokenizer.texts_to_sequences(X_test) 
X_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

word_index = tokenizer.word_index

def create_embedding_layer(word_index, MAX_SEQUENCE_LENGTH):
    EMBEDDING_DIM = 300

    np.random.seed()
    # load embeddings into a dict
    embeddings_index = load_glove()
    
 
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
    return embedding_layer

def lstmModel(multiplier):
    model = Sequential()
    model.add(create_embedding_layer(word_index, MAX_SEQUENCE_LENGTH))
    model.add(LSTM(int(64*multiplier), return_sequences=True, activation="relu"))
    model.add(Dropout(0.3))
    model.add(LSTM(int(64*multiplier), activation="relu"))
    model.add(Dropout(0.3)) 
    model.add(Dense(int(32*multiplier)))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    return model

def rnnModel():
    model = Sequential()
    model.add(create_embedding_layer(word_index, MAX_SEQUENCE_LENGTH))
    model.add(SimpleRNN(int(64), return_sequences=True, activation="relu"))
    model.add(Dropout(0.3))
    model.add(SimpleRNN(int(64), activation="relu"))
    model.add(Dropout(0.3)) 
    model.add(Dense(int(32)))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    return model

def gruModel():
    model = Sequential()
    model.add(create_embedding_layer(word_index, MAX_SEQUENCE_LENGTH))
    model.add(GRU(int(64), return_sequences=True, activation="relu"))
    model.add(Dropout(0.3))
    model.add(GRU(int(64), activation="relu"))
    model.add(Dropout(0.3)) 
    model.add(Dense(int(32)))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    return model

def compile_and_fit(model, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
    # compile, fit & evaluate the model 
    model.compile(loss='categorical_crossentropy',
                optimizer= 'adam',
                metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=( X_dev, Y_dev), batch_size=64, epochs=10)
     
    print("Generate predictions")
    pred = model.predict(X_test)

    print(history.history)
    return history, pred

def evaluation(y_true, pred):
#    pred = model.evaluate(X_test,Y_test)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_pred=pred, y_true = y_true)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_pred=pred, y_true = y_true)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_pred=pred, y_true = y_true)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_pred=pred, y_true = y_true)
    print('F1 score: %f' % f1)

def plot_graph(history):
    
    plt.figure(1)  
     
    # plot the model  
    plt.subplot(211)  
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()
         
    plt.subplot(211)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()

model = lstmModel(multiplier)
print(model.summary())

history, pred = compile_and_fit(model, X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
score = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

plot_graph(history)

model = rnnModel()
print(model.summary())

history, pred = compile_and_fit(model, X_train, Y_train, X_dev, Y_dev, X_test, Y_test) 
score = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

plot_graph(history)

model = gruModel()
print(model.summary())

history, pred = compile_and_fit(model, X_train, Y_train, X_dev, Y_dev, X_test, Y_test) 
score = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

plot_graph(history)

