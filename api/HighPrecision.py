import pandas as pd
import numpy as np
import gensim.downloader
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import string
import re
from keras.utils.data_utils import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, GRU
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import time
def text_to_word_list(text):
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()
    return text


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
class PrecisionService:
    def __init__(self):
        print("------------ PRECISION SERVICE ------------")

        self.train_df = None

        self.word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
        self.stops = None
        self.vocabulary = None
        self.inverse_vocabulary = None

        self.questions_cols = ["question1", "question2"]

        self.max_seq_length = None
        self.embedding_dim = self.word2vec.vector_size
        self.embeddings = None
        self.model = None

        self.load_data()

    def load_data(self):
        self.train_df = pd.read_csv('./data/quora_duplicate_questions.tsv', sep="\t")
        self.stops = set(stopwords.words('english'))
        self.build_vocabulary()
        # self.define_embeddings()
        self.max_seq_length = max(self.train_df.question1.map(lambda x: len(x)).max(),
                                  self.train_df.question2.map(lambda x: len(x)).max())
        self.build_and_load_model()

    def build_vocabulary(self):
        self.vocabulary = dict()
        self.inverse_vocabulary = ['<unk>']

        questions_cols = ['question1', 'question2']

        for index, row in self.train_df.iterrows():
            for question in self.questions_cols:
                word_indices = []
                for word in text_to_word_list(row[question]):


                    if word not in self.vocabulary:
                        self.vocabulary[word] = len(self.inverse_vocabulary)
                        word_indices.append(len(self.inverse_vocabulary))
                        self.inverse_vocabulary.append(word)
                    else:
                        word_indices.append(self.vocabulary[word])

                self.train_df.at[index, question] = word_indices

    def define_embeddings(self):
        self.embeddings = 1 * np.random.randn(len(self.vocabulary) +1, self.embedding_dim)  # This will be the embedding matrix
        self.embeddings[0] = 0  # So that the padding will be ignored

        for word, index in self.vocabulary.items():
            if word in self.word2vec:
                self.embeddings[index] = self.word2vec.get_vector(word)

    def build_and_load_model(self):
        n_hidden = 50
        gradient_clipping_norm = 1.25
        self.define_embeddings()
        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')

        embedding_layer = Embedding(len(self.embeddings) , self.embedding_dim, weights=[self.embeddings] ,input_length=self.max_seq_length, trainable=False)

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        shared_lstm = LSTM(n_hidden)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        malstm = Model([left_input, right_input], [malstm_distance])
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
        malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        malstm.load_weights('./uros_my_best_model_lstm_50.hdf5')

        self.model = malstm
    def question_to_sequence(self, question):
        q2n = []  # q2n -> question numbers representation

        for word in text_to_word_list(question):
            if word in self.stops and word not in self.word2vec.key_to_index:
                continue

            if word in self.vocabulary:
                q2n.append(self.vocabulary[word])
            else:
                q2n.append(0)

        return q2n

    def get_most_similar_question(self, questions, question):
        asked_question_repeated = np.repeat(question, len(questions))
        asked_question_sequence = [self.question_to_sequence(q) for q in asked_question_repeated]
        questions_sequences = [self.question_to_sequence(q) for q in questions]

        asked_question_sequence = pad_sequences(asked_question_sequence, maxlen=self.max_seq_length)
        questions_sequences = pad_sequences(questions_sequences, maxlen=self.max_seq_length)

        predictions = self.model.predict([asked_question_sequence, questions_sequences]).flatten()
        indices = np.argsort(predictions, axis=0)[::-1]
        index = indices[0]

        return questions[index], predictions[index]
