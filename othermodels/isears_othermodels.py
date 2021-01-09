"""
Created on Oct 19, 2020
CS594: Deep Learning for Natural Language Processing
University of Illinois at Chicago
Fall 2020
Project
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout, Embedding, LSTM, GRU, SimpleRNN
import pandas as pd
import string
import re
import spacy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
spacy_nlp = spacy.load('en_core_web_sm')

unigrams = {}
#Load csv
data_df = pd.read_csv('dataset/ISEAR.csv',usecols=['Text','Emotion'])

# Function to load the glove embeddings
# Returns: embeddings_dict (dict)
# Where, embeddings_dict (dict) is a dict of words as keys & the vector as values 
def load_glove():
    embeddings_dict= {}
    with open("Embeddings/glove.6B.300d.txt", 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Function to count all the unigrams
# Arguments:
# tokens: A list of tokens that the document is split into
# Returns: unigrams (dict)
# Where, unigrams (dict) is a dict with the key as the word and the value as the number of occurences 
def countUnigrams(word):
    # finding unique words
    unigrams[word] = unigrams.get(word, 0)+1
    return unigrams

# Function to split a document into a list of tokens
# Arguments:
# doc: A string containing input document
# Returns: tokens (list)
# Where, tokens (list) is a list of tokens that the document is split into
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

labels = data_df['Emotion'].unique()
print(labels)

X_train, X_test, Y_train, Y_test = train_test_split(data_df['Text'], data_df['Emotion'], test_size=0.2, random_state=1)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

Y_train = pd.get_dummies(Y_train)
Y_test = pd.get_dummies(Y_test)
Y_dev = pd.get_dummies(Y_dev)

data = preprocess(data_df['Text'])

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_df['Text']) 

multiplier = 1
# padding sequences to ensure all rows are of equal length
MAX_SEQUENCE_LENGTH = max([len(s) for s in data])
print(MAX_SEQUENCE_LENGTH)

sequences = tokenizer.texts_to_sequences(X_train) 
X_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

sequences = tokenizer.texts_to_sequences(X_dev) 
X_dev = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

sequences = tokenizer.texts_to_sequences(X_test) 
X_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

word_index = tokenizer.word_index

def create_embedding_layer(word_index):
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
                            trainable=False)
    return embedding_layer

def compile_and_fit(model, X_train, Y_train, X_dev, Y_dev, X_test):
    # compile, fit & evaluate the model 
    model.compile(loss='categorical_crossentropy',
                optimizer= 'adam',
                metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=( X_dev, Y_dev), batch_size=64, epochs=20)
      
    print("Generate predictions")
    pred = model.predict(X_test)
    
    print(history.history)
    return history, pred

def lstmModel():
    model = Sequential()
    model.add(create_embedding_layer(word_index))
    model.add(LSTM(128))
    model.add(Dropout(0.3)) 
    model.add(Dense(len(labels), activation='softmax'))
    print(model.summary())
    return model

def rnnModel():
    model = Sequential()
    model.add(create_embedding_layer(word_index))
    model.add(SimpleRNN(256, activation="relu",return_sequences=True))
    model.add(Dropout(0.3))
    model.add(SimpleRNN(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(len(labels), activation='softmax'))
    return model

def gruModel():
    model = Sequential()
    model.add(create_embedding_layer(word_index))
    model.add(GRU(128))
    model.add(Dropout(0.3)) 
    model.add(Dense(len(labels), activation='softmax'))
    print(model.summary())
    return model


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
