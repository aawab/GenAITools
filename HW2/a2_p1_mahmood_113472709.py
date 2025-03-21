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
    def __init__(self, data):
        self.otalTokens=0
        self.vocab=set()

        self.unigramCounts=collections.defaultdict(int)
        self.bigramCounts=collections.defaultdict(int)
        self.trigramCounts=collections.defaultdict(int)
    
    def train(self, datasets):
        pass
    def nextProb(self, history_toks, next_toks):
        pass



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