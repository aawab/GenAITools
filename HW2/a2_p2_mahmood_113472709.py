import random, os, sys, math, csv, re, collections, string
import numpy as np
import csv
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
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

# Checkpoint 2.2
class RecurrentLM(nn.Module):

    def __init__(self, vocab_size, embed_dim, rnn_hidden_dim):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)  

        self.gru = nn.GRU(embed_dim, rnn_hidden_dim, batch_first=True)

        self.layer_norm = nn.LayerNorm(rnn_hidden_dim)

        self.fc = nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, x):

        #input: x: tensor of shape (batch_size, seq_len)

        #output: logits: output of the model.

        #        hidden_state: hidden state of GRU after processing x (sequence of tokens)

        # Order of layers Embedding -> GRU-> Layer norm -> Fully-connected.
        embedding = self.embed(x) 

        output, hidden_state = self.gru(embedding)

        normalized = self.layer_norm(output)

        logits = self.fc(normalized)

        return logits, hidden_state

   #def stepwise_forward(self, x, prev_hidden_state):

        #input: x: tensor of shape (seq_len)

        #       hidden_state: hidden state of GRU after processing x (single token)

        #<FILL IN at Part 2.4>

        #return logits, hidden_state

# Checkpoint 2.3
def trainLM(model, data, pad_token_id, learning_rate, device):

    #input: model - instance of RecurrentLM to be trained

    #       data - contains X and y as defined in 2.1

    #       pad_token_id - tokenizer’s pad token ID for filtering out pad tokens

    #       learning_rate

    #       device - whether to train model on CPU (="cpu") or GPU (="cuda")

    #output: losses - a list of loss values on the train data from each epoch

    

   return losses

# Checkpoint 2.4
def generate(model, tokenizer, start_phrase, max_len, device):

    #input: model - trained instance of RecurrentLM

    #       tokenizer

    #       start_phrase - string containing input word(s)

    #       max_len - max number of tokens to generate

    #       device - whether to inference model on CPU (="cpu") or GPU (="cuda")

    #output: generated_tokens - list of generated token IDs



   return generated_tokens

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

    # Var to hold checkpoint 2.1 song(ENchanted) chunked tensors
    enchantedTensors = None

    # Prepare the dataset
    stackedChunks = []

    for _, row in enumerate(data):
        lyrics = row[2]

        # Remove section markers([Bridge], etc.)
        lyrics = re.sub(r'\n\[[\x20-\x7f]+\]', '', lyrics)

        # Tokenize
        tokenIDs = gpt2Tokenizer.encode(lyrics)

        # Chunk with helper function
        chunks = chunk_tokens(tokenIDs, gpt2Tokenizer.bos_token_id, gpt2Tokenizer.eos_token_id, gpt2Tokenizer.pad_token_id, chunk_len=64)
        if row[0] == "Enchanted (Taylor's Version)":
            enchantedTensors = chunks
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
    print(f"Chunked Tensors for \"Enchanted(Taylor's Version)\": {enchantedTensors}")

    # Checkpoint 2.2
    print(f"logits shape: (batch_size, seq_len, vocab_size)")
    print(f"hidden_state shape: (1, batch_size, rnn_hidden_dim)")
    

    # Checkpoint 2.3

    # Initialzie RecurrentLM
    vocab_size = GPT2TokenizerFast.get_vocab(gpt2Tokenizer).keys()
    embed_dim = 64
    rnn_hidden_dim = 1024
    model = RecurrentLM(vocab_size=vocab_size, embed_dim=embed_dim, rnn_hidden_dim=rnn_hidden_dim)

    # Train LM
    losses = trainLM(model, dataset,gpt2Tokenizer.pad_token_id, 0.0007, device)

    # Plot loss curves
    # TODO: IMPLEMENT TRAINLM AND USE IT HERE ALONGSIDE SETTING UP PLOT LOSS CURVES AND THE REST
    # Use get_perplexity from checkpoint 1.3 to calculate perplexity of the model on below samples

    samples = ["And you gotta live with the bad blood now",
    "Sit quiet by my side in the shade",
    "And I'm not even sorry, nights are so starry",
    "You make me crazier, crazier, crazier, oh",
    "When time stood still and I had you"]

    # Print perplexities and compare with results from 1.3. how does RNN LM perform compared to TrigramLM? Why?


    # Checkpoint 2.4