import random, os, sys, math, csv, re, collections, string
import numpy as np
import csv #(in-built and lightweight!)
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import transformers
from transformers import GPT2TokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import heapq
import matplotlib

# Checkpoint 1.2
class TrigramLM:
    def __init__(self):
        self.totalTokens=0
        self.vocab = GPT2TokenizerFast.get_vocab(GPT2TokenizerFast.from_pretrained('gpt2')).keys()
        
        self.unigramCounts=collections.defaultdict(int)
        self.bigramCounts=collections.defaultdict(int)
        self.trigramCounts=collections.defaultdict(int)
    
    def train(self, datasets):
        # Assume dataset is list of tokensf of all songs

        # Count unigrams, bigrams, and trigrams for all docs
        for doc in datasets:
            doc = doc  # Add start and end tokens
            self.totalTokens += len(doc)

            # Count unigrams
            for i in range(len(doc)):
                token = doc[i]
                self.unigramCounts[token] += 1
            
            # Count bigrams
            for i in range(len(doc)-1):
                bigram = (doc[i], doc[i+1])
                self.bigramCounts[bigram] += 1

            # Count trigrams
            for i in range(len(doc)-2):
                trigram = (doc[i], doc[i+1], doc[i+2])
                self.trigramCounts[trigram] += 1
            
    def nextProb(self, history_toks, next_toks):
        probsList = []

        # Handle OOV tokens
        for i in range(len(history_toks)):
            if history_toks[i] not in self.vocab:
                history_toks[i] = "<OOV>"
        
        # Work for size >2 of history_toks
        if len(history_toks) >= 2:
            h1 = history_toks[-1]
            h2 = history_toks[-2]
        # Work for size 1 of history_toks
        elif len(history_toks) == 1:
            h1 = history_toks[0]
            h2 = "<s>"
        # Work with no hisory_toks
        else:
            h1 = "<s>"
            h2 = "<s>"
        
        V = len(self.vocab)

        # Calculate the probability of next_toks given history_toks
        for next_tok in next_toks:
            if next_tok not in self.vocab:
                next_tok = "<OOV>"
            
            # Calculate the probability
            if len(history_toks) >= 2:
                prob = (self.trigramCounts[(h2, h1, next_tok)] + 1) / (self.bigramCounts[(h2, h1)] + V)
            else:
                # If len(history_toks) <2 then use unigram, skip bigram
                prob = (self.unigramCounts[next_tok] + 1) / (self.totalTokens + V)
            probsList.append((next_tok,prob))
        return probsList

# Checkpoint 1.3
def get_perplexity(probs):

    #input: probs: a list containing probabilities of the target token for each index of input

    #output: perplexity: a single float number

    n = len(probs)
    if n==0 :
        return float('inf')
    
    # Calculate perplexity as inverse of geo mean over the probs
    geoMean = 1
    for prob in probs:
        geoMean *= prob
    perplexity = (1 / geoMean)**(1/n)

    return perplexity

if __name__ == "__main__":

    # Initialize GPT2Tokenizer
    gpt2Tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Add special tokens
    gpt2Tokenizer.pad_token = "<|endoftext|>"
    gpt2Tokenizer.bos_token = "<s>"
    gpt2Tokenizer.eos_token = "</s>"

    # Load the data
    with open('songs.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)[1:-5]
    
    # Checkpoint 1.1

    print("\nCheckpoint 1.1:")
    firstRow = data[0][2]
    lastRow = data[-1][2]

    firstRowTokens = gpt2Tokenizer.tokenize(firstRow)
    lastRowTokens = gpt2Tokenizer.tokenize(lastRow)

    # Print the tokens of first and last rows 
    print(f"first: {firstRowTokens}")
    print(f"last: {lastRowTokens}")

    # Checkpoint 1.2
    print("\nCheckpoint 1.2:")

    # TODO:Q6. The transformer's tokenizer does not automatically add <s> and </s> before and after the document even if BOS and EOS tokens
    #  are set, right? (Output at checkpoint 1.1 says that). In 1.2 we have to add them manually while passing to the tokenizer; is that 
    # correct? (Confusion came to my mind from #55 where I saw: "(1379 > 1024)"; in my case, it is 1381 instead while passing BOS and EOS.)
    # A6. Right, it doesn't automatically add it when you tokenize a sentence. In 1.2, yes, you need to add them manually when training 
    # TrigramLM. Note - The checkpoint test cases in 1.2 and 1.3 can be processed as-is.

    # Initialize the TrigramLM
    trigramLM = TrigramLM()
    trigramLM.train([gpt2Tokenizer.tokenize('<s>'+row[2]+'</s>') for row in data])

    history_toks=['<s>', 'Are', 'Ġwe']
    next_toks=['Ġout', 'Ġin', 'Ġto', 'Ġpretending', 'Ġonly']

    probs = trigramLM.nextProb(history_toks, next_toks)
    print(f"\nhistory: {history_toks}")
    for w, prob in probs:
        print(f"\t{w}: {prob:.5f}")

    history_toks=['And', 'ĠI']
    next_toks=['Ġwas', "'m", 'Ġstood', 'Ġknow', 'Ġscream', 'Ġpromise']

    probs = trigramLM.nextProb(history_toks, next_toks)
    print(f"\nhistory: {history_toks}")
    for w, prob in probs:
        print(f"\t{w}: {prob:.5f}")

    # Checkpoint 1.3
    print("\nCheckpoint 1.3:")

    cases = [['And', 'Ġyou', 'Ġgotta', 'Ġlive', 'Ġwith', 'Ġthe', 'Ġbad', 'Ġblood', 'Ġnow'],
        ['Sit', 'Ġquiet', 'Ġby', 'Ġmy', 'Ġside', 'Ġin', 'Ġthe', 'Ġshade'],
        ['And', 'ĠI', "'m", 'Ġnot', 'Ġeven', 'Ġsorry', ',', 'Ġnights', 'Ġare', 'Ġso', 'Ġstar', 'ry'],
        ['You', 'Ġmake', 'Ġme', 'Ġcraz', 'ier', ',', 'Ġcraz', 'ier', ',', 'Ġcraz', 'ier', ',', 'Ġoh'],
        ['When', 'Ġtime', 'Ġstood', 'Ġstill', 'Ġand', 'ĠI', 'Ġhad', 'Ġyou']]
    
    for case in cases:
        # Calculate token probs for seq
        probs = []
        history = []

        for token in case:
            prob = trigramLM.nextProb(history, [token])
            probs.append(prob[0][1])

            # Update history
            history.append(token)
            if len(history) > 2:
                history = history[-2:]

        perplexity = get_perplexity(probs)
        print(f"Case: {' '.join(case)}")
        print(f"Perplexity: {perplexity:.5f}")

        # TODO: write/print 2-4 line observations about results, why similar or diff, one reason for it?
        # TODO: make usre to append <s> and </s> to the start and end of the case respectively