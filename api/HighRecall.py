# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import spacy
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from scipy.spatial import distance
import numpy as np
#import matplotlib.pyplot as plt
from nltk.tokenize import WhitespaceTokenizer
import re
import time


from sklearn.metrics.pairwise import cosine_distances
from HighPrecision import PrecisionService

class HighRecallService:
    def __init__(self):
        print("------------ HIGH RECALL SERVICE ------------")

        df = pd.read_csv('./data/Q&A.csv', delimiter='\t')
        df = df.rename(columns={"Unnamed: 0": "ID"})
        #self.train_df = None
        self.corpus = df.Question.drop_duplicates().to_numpy()
        self.model = None
        self.wv = None
        self.questions_vec = None
        self.nlp = spacy.load('en_core_web_lg')
        self.load_data()

        self.answers = df.Answer.to_numpy()
    def load_data(self):
        #all_questions = df.Question.to_numpy()
        self.model = Word2Vec.load("word2vec.model")
        self.wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
        self.questions_vec = self.get_document_vectors(self.corpus)

    def get_answer(self, index):
        return self.answers[index]

    def get_document_vectors_pos(self, questions, avg=0):

        questions_vec = []
        start_time1 = time.time()
        for question in questions:
            tokens = self.nlp(question.replace("?", "").replace(".", "").lower())
            question_vec = np.zeros(100, dtype=float)
            for token in tokens:
                try:
                    word_vec = self.wv[token.text]
                except:
                    word_vec = np.zeros(100, dtype=float)


                coef = 1
                if token.pos_ == 'VERB' or token.pos_ == 'PROPN':
                    coef = 1.4
                elif token.pos_ == 'ADV':
                    coef = 1.1
                elif token.pos_ == 'ADJ':
                    coef = 1.3
                elif token.pos_ == 'ADP' or token.pos_ == 'AUX':
                    coef = 0.8
                elif token.pos_ == 'CONJ' or token.pos_ == 'DET':
                    coef = 0.5
                elif token.pos_ == 'INTJ':
                    coef = 0.3

                question_vec += word_vec * coef
            if avg == 1:
                question_vec = question_vec / len(tokens)
            questions_vec.append(question_vec)
        print("time elapsed1: {:.2f}s".format(time.time() - start_time1))
        return questions_vec

    def get_best_n(self, question, N=100):
        test_question_vec = self.get_document_vectors([question])
        distances_array = cosine_distances(test_question_vec, self.questions_vec)[0]
        question_ind = np.argsort(distances_array)[0:N]
        topQuestions = []
        for i in question_ind:
            topQuestions.append([i, self.corpus[i]])
        return topQuestions

    def get_document_vectors(self, questions, avg=0):
        questions_vec = []
        tokenizer = WhitespaceTokenizer()
        for question in questions:
            words = tokenizer.tokenize(question)
            question_vec = np.zeros(100, dtype=float)
            for word in words:
                word = word.replace("?", "").lower()
                try:
                    word_vec = self.wv[word]
                except:
                    word_vec = np.zeros(100, dtype=float)
                question_vec += word_vec
            if avg == 1:
                question_vec = question_vec / len(words)
            questions_vec.append(question_vec)
        return questions_vec