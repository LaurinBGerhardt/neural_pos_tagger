#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python 3.8

from collections import defaultdict, Counter
import sys
import json
from typing import Tuple, List

class Data:
    '''Reads in and formats the data needed for biLSTM pos tagging'''

    UNK = "<UNKNOWN>" #the unknown word

    def __init__(self,*args):
        if len(args)==1:
            self.init_test(*args)   #tagging mode
        else:
            self.init_train(*args)  #training mode

    def init_test(self,filename:str):
        with open(filename,"r",encoding="utf-8") as jsonfile:
            data = json.load(jsonfile)
            self.letter_IDs = defaultdict(int,data["letter_IDs"])
            self.numChars = len(self.letter_IDs)
            self.tag_IDs = defaultdict(int,data["tag_IDs"])
            self.numTags = len(self.tag_IDs)
            self.IDs_tags = defaultdict(lambda:Data.UNK,
                {val:key for key,val in self.tag_IDs.items()})

    def init_train(self,trainfile:str,devfile:str):
        '''Trainfile: filename for test set. Formatting: Word <TAB> POS\n
        Devfile: filename for development set. Same formatting\n
        num Words: Determines size of the vocabulary. All less common words are UNKOWN:0'''
        self.trainSentences = self.read_data(trainfile)
        self.devSentences = self.read_data(devfile)
        # dicts {word/pos : unique ID}:
        self.letter_IDs = defaultdict(int)
        self.tag_IDs = defaultdict(int)

        #Count all words and tags in test set:
        letter_cnt = Counter()
        tag_cnt = Counter()
        for sent,tagged_sent in self.trainSentences:
            for word in sent:
                letter_cnt.update(word)
            tag_cnt.update(tagged_sent)
        self.numTags = len(tag_cnt) #for consistency with numChars the 0 is not counted
        
        #Create defaultdict which gives the numChars most common a unique ID:
        for ID, (char,count) in enumerate(letter_cnt.most_common(), 1):
            if count >= 1:
                self.letter_IDs[char] = ID
        self.numChars = len(self.letter_IDs)
        
        #Create defaultdict which gives all tags in test set a unique ID:
        for ID, (tag,_) in enumerate(tag_cnt.most_common(),1):
            self.tag_IDs[tag] = ID

        self.IDs_tags = defaultdict(lambda:Data.UNK,
            {val:key for key,val in self.tag_IDs.items()})

    def read_data(self,filename:str):
        """Opens file and returns list of format [(sentence,tags)], \n
        where sentence and tags are lists of strings. \n
        FILES MUST END WITH TWO EMPTY LINES"""
        sents_tags = []
        words = []
        tags = []
        for line in open(filename,"r",encoding="utf-8"):
            line = line.rstrip()
            if line:
                word, tag = line.split("\t")
                words.append(word)
                tags.append(tag)
            else: #if line == "" (originally "\n")
                if words and tags:
                    sents_tags.append((words,tags))
                words,tags = [],[]
        return sents_tags

    def words2IDvecs(self,words:list) -> Tuple[List[int], List[int]]:
        '''Creates matrices of IDs from list of words. \n
        Returns a letter ID matrix for the prefix (size 10) of word, and one for the suffix \n
        matrix size each |words|x10'''
        prefixes = []
        suffixes = []
        for word in words:
            prefix = word[::-1].rjust(10)[-10:] #reversed prefix
            suffix = word.rjust(10)[-10:]
            #indexing ok because of defaultdicts:
            prefixes.append([self.letter_IDs[letter] for letter in prefix])
            suffixes.append([self.letter_IDs[letter] for letter in suffix])
        return prefixes, suffixes

    def tags2IDs(self,tags:list):
        '''Creates list of IDs from list of tags. \n
        Each tag corresponds to its tag ID in the return'''
        return [self.tag_IDs[tag] for tag in tags]  #indexing ok because of defaultdict

    def IDs2tags(self,IDs:list):
        '''Creates list of tags from list of tag IDs. \n
        Each ID corresponds to its tag in the return'''
        tag_list = [self.IDs_tags[ID] for ID in IDs]    #ok because of defaultdict
        return tag_list

    def store_parameters(self,filename:str):
        '''Saves letter_IDs and tag_IDs to file in JSON format'''
        with open(filename,"w",encoding="utf-8") as outfile:
            json.dump({"letter_IDs":self.letter_IDs,"tag_IDs":self.tag_IDs},
                        outfile,ensure_ascii=False,indent=4)

    def sentences(self,filename:str):
        '''Reads sentences from file, yields wordlist for each sentence'''
        # for line in open(filename,"r",encoding="utf-8"):
        #     line = line.split("\t")
        #     sent=[]
        #     if line[0]: #fist column has test words
        #         sent.append(line)
        #     else:#if line is empty:
        #         yield sent
        #         sent = []
        with open(filename, encoding="utf-8") as file:
            sent = []
            for line in file:
                line = line.rstrip().split("\t")
                if line[0]: #fist column has test words
                    sent.append(line)
                else:#if line is empty:
                    yield sent
                    sent = []
#END CLASS Data

def run_test():
    '''Used for quickly testing that everything works'''
    test = Data(sys.argv[1],sys.argv[2])
    wordIDs = []
    tagIDs = []
    for words,tags in test.trainSentences:
        prefixes,suffixes = test.words2IDvecs(words +["Schubbel"])  #intentionally added unknown word
        tagIDs = test.tags2IDs(tags)
    ids_of_some_tags = test.IDs2tags([0,2,2,3,1,1])
    print(prefixes,"\n",suffixes,"\n", tagIDs,"\n", ids_of_some_tags)
    for sentence in test.sentences(sys.argv[1]):
        print(sentence)

if __name__ == "__main__":
    run_test()

