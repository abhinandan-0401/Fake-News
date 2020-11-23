from preprocess import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

# Keras
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model, model_from_json
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import LSTM, Flatten, Dropout, Activation, Input, Dense, BatchNormalization, Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras import regularizers
from keras.optimizers import Adam, Adadelta, Adamax
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, classification_report)
import logging
logging.basicConfig(level=logging.INFO)

def preprocess(docs, samp_size=None):
    """
    Preprocess the data
    """
    if not samp_size:
        samp_size = len(docs)

    print('Preprocessing raw texts ...')
    n_docs = len(docs)
    sentences = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed
    idx_in = []  # index of sample selected
    #     samp = list(range(100))
    samp = np.random.choice(n_docs, samp_size)
    for i, idx in enumerate(samp):
        sentence = preprocess_sent(docs[idx])
        token_list = preprocess_word(sentence)
        if token_list:
            idx_in.append(idx)
            sentences.append(sentence)
            token_lists.append(token_list)
        print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
    print('Preprocessing raw texts. Done!')
    return sentences, token_lists, idx_in

def check_Classification_Results(y_test, y_pred):
    print("Confusion Matrix : \n", confusion_matrix(y_test,y_pred))
    print("Accuracy Score : \n", accuracy_score(y_test,y_pred))
    print("F1 Score : \n", f1_score(y_test,y_pred))
    print("Precision Score : \n", precision_score(y_test,y_pred))
    print("Recall Score : \n", recall_score(y_test,y_pred))
    print("ROC-AUC Score : \n", roc_auc_score(y_test,y_pred))
    print("\n\n")

def plot_results(history, path, id):
    his_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1,2)
    
    # Plot training vs validation loss
    axes[0].plot(his_df['loss'],label="Training")
    axes[0].plot(his_df['val_loss'],label="Validation")
    axes[0].legend(loc='best')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')

    # Plot training vs validation accuracy
    axes[1].plot(his_df['accuracy'],label="Training")
    axes[1].plot(his_df['val_accuracy'],label="Validation")
    axes[1].legend(loc='best')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')

    # saving the figure. 
    plt.savefig(path + "Training_History_" + id + ".png", 
                bbox_inches ="tight", 
                pad_inches = 1, 
                transparent = True, 
                orientation ='landscape') 

# define model object   
class Sequence_Classification():

    def __init__(self, n_classes, embed_size, maxlen):

        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        
        self.id = "TomTom" + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.n_classes = n_classes
        self.vocab_size = None
        self.embed_size = embed_size
        self.maxlen = maxlen
        self.tokenizer = None
        self.model_dir = None
        self.model = None
    
    def tokenize(self,X):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.word_index) + 1

    
    def get_Model_Info(self,):
    
        lstm_layers = 2
        n_lstms = 200
        dropout_rate = 0.4
        n_hidden = 3
        n_neurons = 256
        act_fn = 'relu'
        k_init = 'glorot_normal'
        lr = 0.00001
        
        return lstm_layers, n_lstms, dropout_rate, n_hidden, n_neurons, act_fn, k_init, lr

    def get_Output_Layer(self,):
        if self.n_classes == 2:
            op_units = 1
            op_act = 'sigmoid'
            loss_fn = 'binary_crossentropy'
        else:
            op_act = 'softmax'
            loss_fn = 'sparse_categorical_crossentropy'
        return op_units, op_act, loss_fn

    # simple Callbacks
    def getCallbacks(self, patience, verbose):
        es = EarlyStopping(monitor='val_loss', 
                        mode='min', 
                        min_delta=0.001,
                        verbose=verbose, 
                        patience=patience)
        mc = ModelCheckpoint(self.model_dir + 'best_weights_' + self.id + '.h5', 
                            monitor='val_accuracy', 
                            mode='max', 
                            verbose=verbose, 
                            save_best_only=True)
        return [es,mc]


    
    def compile_nn(self,):
        """
        compile the computational graph
        
        """

        lstm_layers, n_lstms, dropout_rate, n_hidden, n_neurons, act_fn, k_init, lr = self.get_Model_Info()
        
        model = Sequential(name=self.id)
        model.add(Embedding(self.vocab_size, 
                            self.embed_size, 
                            input_length=self.maxlen))

        for _ in range(lstm_layers):
            model.add(BatchNormalization())
            lstm = LSTM(n_lstms,
                        return_sequences=True,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate)
            model.add(lstm)

        model.add(Flatten())

        for _ in range(n_hidden):
            model.add(BatchNormalization())
            dense = Dense(n_neurons, activation=act_fn, kernel_initializer=k_init)
            model.add(dense)

        op_units, op_act, loss_fn = self.get_Output_Layer()
        model.add(Dense(op_units, activation=op_act))

        # define optimizer
        opt = keras.optimizers.Adam(learning_rate=lr)

        # compile the model
        model.compile(optimizer=opt, 
                    loss=loss_fn, 
                    metrics=['accuracy'])
        
        print(model.summary())
        self.model = model
        
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.model_dir + "model" + self.id + ".json", "w") as json_file:
            json_file.write(model_json)

    def fit(self, X_train, y_train):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        self.tokenize(X_train)
        
        if not self.model:
            self.compile_nn()

        callback_list = self.getCallbacks(20, 1)

        trainsequences = pad_sequences(self.tokenizer.texts_to_sequences(X_train), 
                                    maxlen=self.maxlen, 
                                    padding='post')

        history = self.model.fit(trainsequences,
                                y_train,
                                batch_size=32,
                                epochs=100,
                                verbose=1,
                                callbacks=callback_list,
                                validation_split=0.2,
                                shuffle=True).history
        print(classification_report(y_train,self.model.predict_classes(trainsequences)))
        plot_results(history,self.model_dir,self.id)

    def re_fit(self, X_train, y_train):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        _, _, _, _, _, _, _, lr = self.get_Model_Info()
        _, _, loss_fn = self.get_Output_Layer()

        # define optimizer
        opt = keras.optimizers.Adam(learning_rate=lr)
        
        callback_list = self.getCallbacks(20, 1)

        # load json and create model
        json_file = open(self.model_dir + 'model' + self.id + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.model_dir + 'best_weights_' + self.id + '.h5')

        self.id = "TomTom" + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # compile the model
        self.model.compile(optimizer=opt, 
                        loss=loss_fn, 
                        metrics=['accuracy'])

        trainsequences = pad_sequences(self.tokenizer.texts_to_sequences(X_train), 
                                    maxlen=self.maxlen, 
                                    padding='post')

        history = self.model.fit(trainsequences,
                                y_train,
                                batch_size=32,
                                epochs=100,
                                verbose=1,
                                callbacks=callback_list,
                                validation_split=0.2,
                                shuffle=True).history

        print(classification_report(y_train,self.model.predict_classes(trainsequences)))
        plot_results(history,self.model_dir,self.id)

    def predict(self, X_test):
        """
        Predict topics for new_documents
        """

        # load json and create model
        json_file = open(self.model_dir + 'model' + self.id + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.model_dir + 'best_weights_' + self.id + '.h5')

        testsequences = pad_sequences(self.tokenizer.texts_to_sequences(X_test), 
                                    maxlen=self.maxlen, 
                                    padding='post')
        
        test_pred = self.model.predict_classes(testsequences)
        return test_pred
