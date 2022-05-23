#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python 3.8

from Data import Data
import torch

import argparse

from TaggerModel import TaggerModel
argparser = argparse.ArgumentParser()
argparser.add_argument("paramfile",help="File (w/out file ending) to read neural network from")
argparser.add_argument("testfile",help="Filename of testing set")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def output(sentence, tags):
    '''Prints the true tags on the left and the predicted on the right, prettily'''
    space = " "*30
    for t,p in zip(sentence,tags):
        print(t,space[len(t):],p)
    print("")

def main():
    '''Creates Data object using JSON file, \n
    Creates TaggerModel object using <paramfile>.rnn file'''
    args = argparser.parse_args()
    data = Data(args.paramfile+".io")
    # model = torch.load(args.paramfile+".rnn").to(DEVICE) # appears to be depricated
    # model = TaggerModel.load_state_dict(torch.load(args.paramfile+".rnn")).to(DEVICE)
    model = TaggerModel.from_argfile().to(DEVICE)
    model.load_state_dict(torch.load(args.paramfile+".pth"))
    model.train(False)

    with torch.no_grad():
        for sentence in data.sentences(args.testfile):
            pref,suff = data.words2IDvecs(sentence)
            prefixes = torch.LongTensor(pref).to(DEVICE)
            suffixes = torch.LongTensor(suff).to(DEVICE)
            pred_tagIDs = torch.argmax(model(prefixes,suffixes),dim=1).tolist() # dim 0 is sent length
            tags = data.IDs2tags(pred_tagIDs)
            #print(sentence)
            #print(tags)
            output(sentence, tags)
            
    return data,model

if __name__ == "__main__":
    main()

