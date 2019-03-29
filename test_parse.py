# -*- coding: utf-8 -*-

"""
parserを使う
"""

from Summarizer import Summarizer
from Parser import Parser

parser = Parser()

words = ['英国', '0.9']


parse_history = {}

for i in range(len(words)):
    w = words[i]
    key, source = parser.parse_source(w)
    if key != "":
        if key not in parse_history:
            newList = []
            newList.append(source)
            parse_history[key] = newList
        else:
            preList = parse_history[key]
            preList.append(source)
            parse_history[key] = preList
    words[i] = parser.parse(w)
words = " ".join(words)
words = parser.parse_num_fix(words)

words = words.split(' ')

encoder_decoder_model = Summarizer()
modelfile = "models/energy-paper-summarizer-20181118-13.model"
encoder_decoder_model.load_model(modelfile)

reply = encoder_decoder_model.test(words)

reply = "".join(reply)

for key in parse_history:
    sourceList = parse_history[key]
    for source in sourceList:
        if reply.find(key) == -1:
            break
        reply = reply.replace(key, source, 1)

print(reply)