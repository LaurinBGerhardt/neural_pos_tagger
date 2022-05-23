#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python 3.8

from Data import Data
import torch

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("paramfile",help="File (w/out file ending) to read neural network from")
argparser.add_argument("testfile",help="Filename of testing set")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    '''Creates Data object using JSON file, \n
    Creates TaggerModel object using <paramfile>.rnn file'''
    args = argparser.parse_args()
    data = Data(args.paramfile+".io")
    model = torch.load(args.paramfile+".rnn").to(DEVICE)
    model.train(False)

    with torch.no_grad():
        for sentence in data.sentences(args.testfile):
            pref,suff = data.words2IDvecs(sentence)
            prefixes = torch.LongTensor(pref).to(DEVICE)
            suffixes = torch.LongTensor(suff).to(DEVICE)
            pred_tagIDs = torch.argmax(model(prefixes,suffixes),dim=1).tolist() # dim 0 is sent length
            tags = data.IDs2tags(pred_tagIDs)
            print(sentence)
            print(tags)
    return data,model

if __name__ == "__main__":
    main()

