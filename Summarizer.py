# -*- coding: utf-8 -*-

"""
要約文生成モデル
attention-encoder-decoderモデル
"""

import numpy as np
import chainer
from chainer import Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L

class Summarizer(chainer.Chain):
    def __init__(self,
                 vocab_source = 'texts/energy_paper_wakati_neologd.txt',
                 source = 'texts/words_20181118_parse.txt',
                 target = 'texts/sentence_20181118_parse.txt',
                 new_source = '',
                 new_target = '',
                 embed_size = 200):
        self.embed_size = embed_size
        self.source_lines = self.load_language(source)
        self.target_lines = self.load_language(target)
        self.word2id, self.id2word = self.make_vocab(vocab_source, source, target)
        if new_source != '' and new_target != '':
            self.new_source_lines = self.load_language(new_source)
            self.new_target_lines = self.load_language(new_target)
        
        vocab_size = len(self.word2id)
        super(Summarizer, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            H       = L.LSTM(embed_size, embed_size),
            Wc1     = L.Linear(embed_size, embed_size),
            Wc2     = L.Linear(embed_size, embed_size),
            W       = L.Linear(embed_size, vocab_size),
        )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)
    
    def learn(self):
        line_num = len(self.source_lines) - 1
        for i in range(line_num):
            source_words = self.source_lines[i].split()
            target_words = self.target_lines[i].split()

            self.H.reset_state()
            self.zerograds()        
            loss = self.loss(source_words, target_words)
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()
    
    def learn_new(self):
        line_num = len(self.new_source_lines) - 1
        for i in range(line_num):
            source_words = self.new_source_lines[i].split()
            target_words = self.new_target_lines[i].split()

            self.H.reset_state()
            self.zerograds()        
            loss = self.loss(source_words, target_words)
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()
    
    def test(self, source_words):
        bar_h_i_list = self.h_i_list(source_words, True)
        x_i = self.embed(Variable(np.array([self.word2id['<eos>']], dtype=np.int32), volatile='on'))
        h_t = self.H(x_i)
        c_t = self.c_t(bar_h_i_list, h_t.data[0], True)

        result = []
        bar_h_t = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
        wid = np.argmax(F.softmax(self.W(bar_h_t)).data[0])
        result.append(self.id2word[wid])

        loop = 0
        while (wid != self.word2id['<eos>']) and (loop <= 30):
            y_i = self.embed(Variable(np.array([wid], dtype=np.int32), volatile='on'))
            h_t = self.H(y_i)
            c_t = self.c_t(bar_h_i_list, h_t.data, True)

            bar_h_t = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
            wid = np.argmax(F.softmax(self.W(bar_h_t)).data[0])
            result.append(self.id2word[wid])
            loop += 1
        return result
    
    # 損失を求める
    def loss(self, source_words, target_words):
        bar_h_i_list = self.h_i_list(source_words)
        x_i = self.embed(Variable(np.array([self.word2id['<eos>']], dtype=np.int32)))
        h_t = self.H(x_i)
        c_t = self.c_t(bar_h_i_list, h_t.data[0])

        bar_h_t    = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
        tx         = Variable(np.array([self.word2id[target_words[0]]], dtype=np.int32))
        accum_loss = F.softmax_cross_entropy(self.W(bar_h_t), tx)
        for i in range(len(target_words)):
            wid = self.word2id[target_words[i]]
            y_i = self.embed(Variable(np.array([wid], dtype=np.int32)))
            h_t = self.H(y_i)
            c_t = self.c_t(bar_h_i_list, h_t.data)

            bar_h_t    = F.tanh(self.Wc1(c_t) + self.Wc2(h_t))
            next_wid   = self.word2id['<eos>'] if (i == len(target_words) - 1) else self.word2id[target_words[i+1]]
            tx         = Variable(np.array([next_wid], dtype=np.int32))
            loss       = F.softmax_cross_entropy(self.W(bar_h_t), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss
        
    def load_language(self, filename):
        lines = open(filename, encoding="utf-8").read().split('\n')
        return lines
    
    def make_vocab(self, filename1, filename2, filename3):
        word2id = {}
        lines = open(filename1, encoding="utf-8").read().split('\n')
        for i in range(len(lines)):
            sentence = lines[i].split()
            for word in sentence:
                if word not in word2id:
                    word2id[word] = len(word2id)
        lines = open(filename2, encoding="utf-8").read().split('\n')
        for i in range(len(lines)):
            sentence = lines[i].split()
            for word in sentence:
                if word not in word2id:
                    word2id[word] = len(word2id)
        lines = open(filename3, encoding="utf-8").read().split('\n')
        for i in range(len(lines)):
            sentence = lines[i].split()
            for word in sentence:
                if word not in word2id:
                    word2id[word] = len(word2id)
        word2id['<eos>'] = len(word2id)
        id2word = {v:k for k, v in word2id.items()}
        return [word2id, id2word]
    
    # 単語を追加する
    def add_words(self):
        lines = self.new_source_lines
        for i in range(len(lines)):
            sentence = lines[i].split()
            for word in sentence:
                if word not in self.word2id:
                    self.update_vocab(word)
        lines = self.new_target_lines
        for i in range(len(lines)):
            sentence = lines[i].split()
            for word in sentence:
                if word not in self.word2id:
                    self.update_vocab(word)
    
    # self.word2id, self.id2wordに単語を追加、embedIDも更新
    def update_vocab(self, word):
        self.id2word[len(self.word2id)] = word
        self.word2id[word] = len(self.word2id)
        # embedを更新
        add_W = np.random.randn(1, self.embed_size).astype(np.float32)
        self.embed.W.data = np.r_[self.embed.W.data, add_W]
    
    # h_i のリストを求める
    def h_i_list(self, words, test = False):
        h_i_list = []
        volatile = 'on' if test else 'off'
        for word in words:
            wid = self.word2id[word]
            x_i = self.embed(Variable(np.array([wid], dtype=np.int32), volatile=volatile))
            h_i = self.H(x_i)
            h_i_list.append(np.copy(h_i.data[0]))
        return h_i_list

    # context vector c_t を求める
    def c_t(self, bar_h_i_list, h_t, test = False):
        s = 0.0
        for bar_h_i in bar_h_i_list:
            s += np.exp(h_t.dot(bar_h_i))

        c_t = np.zeros(self.embed_size)
        for bar_h_i in bar_h_i_list:
            alpha_t_i = np.exp(h_t.dot(bar_h_i)) / s
            c_t += alpha_t_i * bar_h_i
        volatile = 'on' if test else 'off'
        c_t = Variable(np.array([c_t]).astype(np.float32), volatile=volatile)
        return c_t
    
    # モデルを読み込む
    def load_model(self, filename):
        serializers.load_npz(filename, self)

    # モデルを書き出す
    def save_model(self, filename):
        serializers.save_npz(filename, self)