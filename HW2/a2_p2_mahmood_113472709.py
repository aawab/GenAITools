import random, os, sys, math, csv, re, collections, string
import numpy as np
import csv #(in-built and lightweight!)
import torch
from torch import nn, Tensor, DataLoader, TensorDataset
import torch.nn.functional as F

import transformers 
from transformers import GPT2TokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import heapq
import matplotlib

# Checkpoint 2.1
def chunk_tokens(tokens, start_token_id, end_token_id, pad_token_id, chunk_len=128):

    #input: tokens: a list containing token ids

    #       start_token_id, end_token_id, pad_token_id: special token ids from the tokenizer

    #       chunk_len: the length of output sequences

    #output: chunks: torch.tensor of sequences of shape (#chunks_in_song, chunk_len)

    chunks = []
    numChunks = (len(tokens) + chunk_len -3) // (chunk_len - 2)

    for i in range(numChunks):
        # Get tokens
        startID = i * (chunk_len - 2)
        endID = min(startID + chunk_len - 2, len(tokens))
        chunk = tokens[startID:endID]

        # Append BOS and EOS token ids as first and last index of chunk
        chunk = [start_token_id] + chunk + [end_token_id]

        # Fill gap using pad token if not required num of tokens
        if chunk_len - len(chunk) > 0:
            chunk += [pad_token_id] * (chunk_len - len(chunk))
        
        chunks.append(chunk)

    return torch.tensor(chunks)

if __name__ == "__main__":

    # Initialize GPT2Tokenizer
    gpt2Tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Add special tokens
    gpt2Tokenizer.pad_token = "<|endoftext|>"
    gpt2Tokenizer.bos_token = "<s>"
    gpt2Tokenizer.eos_token = "</s>"

    # TODO: doesn't work??? Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    with open('songs.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)[1:-5]

    # Prepare the dataset
    stackedChunks = []

    for _, row in enumerate(data):
        lyrics = row['Lyrics']

        # Remove section markers([Bridge], etc.)
        lyrics = re.sub('\n\[[\x20-\x7f]+\]', '', lyrics)

        # Tokenize
        tokenIDs = gpt2Tokenizer.encode(lyrics)

        # Chunk with helper function
        chunks = chunk_tokens(tokenIDs, gpt2Tokenizer.bos_token_id, gpt2Tokenizer.eos_token_id, gpt2Tokenizer.pad_token_id, chunk_len=64)

        stackedChunks.append(chunks)
    
    # Stack the chunks into shape(#allchunks, chunk_len)
    stackedChunks = torch.cat(stackedChunks)

    # Create the X and y for self-supervised learning
    # X contains all but last column, y excludes first column
    X = stackedChunks[:, :-1]
    y = stackedChunks[:, 1:]

    # Batching dataset w tensordataset and dataloader
    dataset  = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # Checkpoint 2.1
    print("\nCheckpoint 2.1:")
    