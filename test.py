# -*- coding: utf-8 -*-

"""
学習済みモデルを用いた文生成スクリプト
"""

from Summarizer import Summarizer
import MeCab

# summarizerのテスト

# sequence = "国内再生可能エネルギー導入量1993年2004年10倍増加"
# sequence = "1993年2004年国内10倍増加導入量再生可能エネルギー"
# sequence = "2013年度実質GDP成長率2.3%増"
sequence = "ガス料金原価様々要素構成おる比較多様方法ある単純対比困難日本産業用0.38(米ドル/m3)家庭用1.12(米ドル/m3)ガス料金他国米国産業用0.11(米ドル/m3)家庭用0.33(米ドル/m3)英国産業用0.25(米ドル/m3)家庭用0.61(米ドル/m3)フランス産業用0.37(米ドル/m3)家庭用0.79(米ドル/m3)比べる高位ある"

# sequence = "国際的金融危機2009年度悪影響減少生産"

m = MeCab.Tagger("-Owakati")
words = m.parse(sequence).split(' ')
words = words[:-1]

model = Summarizer()
modelfile = "models/energy-paper-random-order-summarizer-20181105-21.model"
model.load_model(modelfile)

reply = model.test(words)
    
for re in reply:
    print(re)