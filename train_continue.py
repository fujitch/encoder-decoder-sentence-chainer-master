# -*- coding: utf-8 -*-

"""
summarizer学習再開スクリプト
"""

from Summarizer import Summarizer
import time

# 英日翻訳の学習

start_time = time.time()
print("model new start.")
pre_epochs = 11

model = Summarizer()
model.load_model("models/energy-paper-random-order-summarizer-20181105-" + str(pre_epochs - 1) + ".model")
model.add_words()

elapsed_time = time.time() - start_time
print("model new finished. elapsed_time: {0:.1f}[sec]".format(elapsed_time))

epoch_num = 100
for epoch in range(epoch_num):
    epoch += pre_epochs
    start_time = time.time()
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))

    # 学習を実施
    # model.learn()
    model.learn_new()
    modelfile = "energy-paper-random-order-summarizer-20181105-" + str(epoch) + ".model"
    model.save_model(modelfile)

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (epoch_num - epoch - 1)
    print("{0} / {1} epoch finished.".format(epoch + 1, epoch_num), end="")
    print(" elapsed_time: {0:.1f}[sec]".format(elapsed_time), end="")
    print(" remaining_time: {0:.1f}[sec]".format(remaining_time))