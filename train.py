# -*- coding: utf-8 -*-

"""
summarizer学習スクリプト
"""

from Summarizer import Summarizer
import time

start_time = time.time()
print("model new start.")

model = Summarizer()

elapsed_time = time.time() - start_time
print("model new finished. elapsed_time: {0:.1f}[sec]".format(elapsed_time))

epoch_num = 100
for epoch in range(epoch_num):
    start_time = time.time()
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))

    # 学習を実施
    model.learn()
    modelfile = "energy-paper-summarizer-20181118-" + str(epoch) + ".model"
    model.save_model(modelfile)

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (epoch_num - epoch - 1)
    print("{0} / {1} epoch finished.".format(epoch + 1, epoch_num), end="")
    print(" elapsed_time: {0:.1f}[sec]".format(elapsed_time), end="")
    print(" remaining_time: {0:.1f}[sec]".format(remaining_time))