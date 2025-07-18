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

    if len(sys.argv) != 2:
        print("Usage: python a2_p1_mahmood_113472709.py <filename>")
        sys.exit(1)
    
    # Grab input file and read in the data
    file = sys.argv[1]

    # Load the data
    with open(file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)[1:-5]
    
    # Initialize GPT2Tokenizer
    gpt2Tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Add special tokens
    gpt2Tokenizer.pad_token = "<|endoftext|>"
    gpt2Tokenizer.bos_token = "<s>"
    gpt2Tokenizer.eos_token = "</s>"

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
    print("\nOBSERVATIONS:\nThe perplexity values here seem very high, but I'd say that's almost to be expected. Some of the tokens are rare, and the model is not trained on a large enough dataset to get a good estimate of the probabilities, nor is it a very intensive/thorough training method(trigram LM). Not only that, but the vocabulary of the tokenizer is extremely large and thus the individual probabilities of a next word occuring given history is very low, leading to larger perplexity values overall. Some sentences have higher perplexity because they use a more unorthodox sequence of words than the previous lyrics the trigramLM was trained on.") 
