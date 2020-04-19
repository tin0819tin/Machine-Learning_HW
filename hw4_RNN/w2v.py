import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec

###
from util import load_training_data, load_testing_data
path_prefix = './'

def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('training_label.txt')
    train_x_no_label = load_training_data('training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('testing_data.txt')

    model = train_word2vec(train_x + train_x_no_label + test_x)
    #model = train_word2vec(train_x + test_x)
    
    print("saving model ...")
    #model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join(path_prefix, 'w2v_all.model'))