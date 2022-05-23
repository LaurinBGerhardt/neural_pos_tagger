#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python 3.8

from Data import Data
from TaggerModel import TaggerModel
import random

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple

import argparse
argparser = argparse.ArgumentParser(description="Trains biLSTM POS-tagger model")
argparser.add_argument("trainfile"      ,help="Filename of training set without file ending")
argparser.add_argument("devfile"        ,help="Filename of development set without file ending")
argparser.add_argument("paramfile"      ,help="File (w/out file ending) to store mapping tables (word2IDs,tag2IDs) to")
argparser.add_argument("--num_epochs"   ,type=int   ,default=20 ,help="How often training set is used")
argparser.add_argument("--emb_size"     ,type=int   ,default=200,help="Size of word embeddings")
argparser.add_argument("--char_rnn_size",type=int   ,default=200,help="Number of LSTM parameters")
argparser.add_argument("--word_rnn_size",type=int   ,default=200,help="Number of biLSTM parameters")
argparser.add_argument("--dropout_rate" ,type=float ,default=0.5)
argparser.add_argument("--learning_rate",type=float ,default=0.001)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def output(true,pred):
    '''Prints the true tags on the left and the predicted on the right, prettily'''
    space = " "*30
    print("true",space[len("true"):],"predicted")
    for t,p in zip(true,pred):
        print(t,space[len(t):],p)

def predict(data:Data,model:TaggerModel)-> Tuple[list, list]:
    '''Uses development set to predict current best-guess of tags. \n
    Returns ([true_tags],[predicted_tags]), each not bundled in sentence-chunks'''
    true_tagIDs=[]
    pred_tagIDs=[]
    with torch.no_grad():
        for sentence, true_tags in data.devSentences:
            true_tagIDs.extend(data.tags2IDs(true_tags))
            pref,suff = data.words2IDvecs(sentence)
            prefixes = torch.LongTensor(pref).to(DEVICE)
            suffixes = torch.LongTensor(suff).to(DEVICE)
            pred_tagIDs.extend(torch.argmax(model(prefixes,suffixes),dim=1).tolist()) # dim 0 is the sent length
    return true_tagIDs,pred_tagIDs

def calculate_accuracy(data:Data,model:TaggerModel) -> float:
    '''Calculates accuracy of model'''
    y_true,prediction = predict(data,model)
    total = len(y_true)
    correct = sum([label==pred for label,pred in zip(y_true,prediction)])
    return 0 if total==0 else correct/total

def main():
    '''Call function with (for example) \n
    ./rnn-train.py trainfile devfile paramfile --num_epochs=20 --num_words=10000 --emb_size=200 --char_rnn_size=200 --word_rnn_size=200 --dropout_rate=0.5 --learning_rate=0.5 > accuracy.txt\n
    Initializes and trains TaggerModel using data from training and development file'''
    # torch.autograd.set_detect_anomaly(True)
    args = argparser.parse_args()
    data = Data(args.trainfile,args.devfile)
    data.store_parameters(args.paramfile+".io")

    model = TaggerModel(
        data.numChars,
        data.numTags,
        args.emb_size,
        args.char_rnn_size,
        args.word_rnn_size,
        args.dropout_rate
        ).to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

    bestever_accuracy = 0.

    for epoch in range(0,args.num_epochs):
        random.shuffle(data.trainSentences)
        #train
        model.train(True)
        for sentence,tags in data.trainSentences:
            model.zero_grad()
            pres,sufs   = data.words2IDvecs(sentence)
            prefixes    = torch.LongTensor(pres)                 .to(DEVICE)
            suffixes    = torch.LongTensor(sufs)                 .to(DEVICE)
            tagIDs      = torch.LongTensor(data.tags2IDs(tags))  .to(DEVICE)
            scores      = model(prefixes,suffixes)               .to(DEVICE)
            # scores      = torch.squeeze(output,dim=0)            .to(DEVICE)
            loss        = loss_func(scores,tagIDs)
            loss.backward()
            optimizer.step()
        model.train(False)
        #if new best accuracy reached, saves current model in paramfile.rnn:
        current_accuracy = calculate_accuracy(data,model)
        if current_accuracy >= bestever_accuracy:
            torch.save(model,args.paramfile+".rnn")
            bestever_accuracy = current_accuracy
    #to be redirected in bash using "> outputfile"
    print("Best accuracy: ",str(bestever_accuracy)) 
    print("Params:")
    print(  "Epochs:",args.num_epochs,
            "\nEmbedding Size:",args.emb_size,
            "\nChar LSTM RNN Size:",args.char_rnn_size,
            "\nWord biLSTM RNN Size:",args.word_rnn_size,
            "\nDropout Rate:",args.dropout_rate,
            "\nLearning Rate:",args.learning_rate)

if __name__ == "__main__":
    main()

