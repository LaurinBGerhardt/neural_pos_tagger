#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python 3.8

import torch
import torch.nn as nn


class TaggerModel(nn.Module):
    '''LSTM-based model for predicting POS tags'''
    
    def __init__(self,numChars:int,numTags:int,char_embSize:int,char_rnn_size:int,word_rnn_size:int,dropoutRate:float):
        '''numWords: Vocabulary size minus the unknown word \n
        numTags: Tag vocab size minus the unknown tag \n
        char_embSize: The number of features for each letter \n
        rnnSize: The number of outputs of one of the LSTMs \n
        dropoutRate: Likelihood of bits randomly being set to 0 to prevent overfitting'''
        super(TaggerModel,self).__init__()
        self.char_embedding = nn.Embedding(numChars+1,char_embSize)
        self.char_forwardLSTM = nn.LSTM(input_size=char_embSize,hidden_size=char_rnn_size,batch_first=True)
        self.char_backwardLSTM = nn.LSTM(input_size=char_embSize,hidden_size=char_rnn_size,batch_first=True)
        self.dropout = nn.Dropout(p=dropoutRate,inplace=False)
        self.wordLSTMs = nn.LSTM(input_size=char_rnn_size*2,hidden_size=word_rnn_size,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(word_rnn_size*2,numTags+1) 

    def forward(self,prefix_IDs:torch.LongTensor,suffix_IDs:torch.LongTensor):
        '''Forward-propagating\n
        prefixes: matrix containing all word-prefixes (len 10) of sentence \n
        suffixes: analogous. \n
        Prefixes and suffixes are tensors of letter-IDs'''
        prefixes = self.char_embedding(prefix_IDs)
        prefixes = self.dropout(prefixes)
        suffixes = self.char_embedding(suffix_IDs)
        suffixes = self.dropout(suffixes)
        # unsqueezed_pre = torch.unsqueeze(prefixes,dim=0)
        # unsqueezed_suf = torch.unsqueeze(suffixes,dim=0)

        prefix_LSTMout, _ = self.char_backwardLSTM(prefixes)
        suffix_LSTMout, _ = self.char_forwardLSTM(suffixes)
        prefix_LSTMout = self.dropout(prefix_LSTMout[:,-1,:])
        suffix_LSTMout = self.dropout(suffix_LSTMout[:,-1,:])

        word_repr = torch.cat((suffix_LSTMout,prefix_LSTMout),dim=1)
        word_repr = torch.unsqueeze(word_repr,dim=0)
        word_lstmout, _ = self.wordLSTMs(word_repr)
        word_lstmout = torch.squeeze(word_lstmout,dim=0)
        word_lstmout = self.dropout(word_lstmout)#
        tag_scores = self.linear(word_lstmout)
        return tag_scores
#END CLASS TaggerModel

def run_test():
    '''Quickly assessing that the dimensions check out'''
    tagger = TaggerModel(10, 3, 4, 4, 4, 0.5)
    prefixes = torch.LongTensor([
        [1,0,2,4,2,1,2,3,0,0],
        [1,1,8,4,2,3,2,5,7,0]
    ])
    suffixes = torch.LongTensor([
        [2,0,7,4,2,1,2,3,0,1],
        [3,6,2,4,5,1,2,3,9,0]
    ])
    tagScores = tagger.forward(prefixes,suffixes)
    print(tagScores)

if __name__ == "__main__":
    run_test()



