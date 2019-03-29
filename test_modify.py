# -*- coding: utf-8 -*-

"""
キーワードが入るように生成文を修正する
"""

from Summarizer import Summarizer
import MeCab
from gensim.models import word2vec
import numpy as np
import pickle

try:
    word2vec_model
except NameError:
    word2vec_model = pickle.load(open('energy_paper_model.pickle', 'rb'))
    

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def detect_word(objectList, keyword):
    for obj in objectList:
        if obj == keyword:
            return True
    return False

def detect_similar_index(objectList, keyword):
    index = 0
    score = 0.0
    keyword_vector = word2vec_model.wv[keyword]
    for i in range(len(objectList)):
        obj = objectList[i]
        vector = word2vec_model.wv[obj]
        if score < cos_sim(keyword_vector, vector):
            score = cos_sim(keyword_vector, vector)
            index = i
    return index

sequence = "2.3%増2013年度実質GDP成長率"
m = MeCab.Tagger("-Owakati")
words = m.parse(sequence).split(' ')
words = words[:-1]

encoder_decoder_model = Summarizer()
modelfile = "models/energy-paper-random-order-summarizer-20181105-9.model"
encoder_decoder_model.load_model(modelfile)

reply = encoder_decoder_model.test(words)

for keyword in words:
    if detect_word(reply[:-1], keyword):
        continue
    reply[detect_similar_index(reply[:-1], keyword)] = keyword

for re in reply:
    print(re)