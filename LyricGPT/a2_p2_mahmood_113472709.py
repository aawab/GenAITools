import random, os, sys, math, csv, re, collections, string
import token
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
import matplotlib.pyplot as plt

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

    def stepwise_forward(self, x, prev_hidden_state):

        #input: x: tensor of shape (seq_len)

        #       hidden_state: hidden state of GRU after processing x (single token)

        #<FILL IN at Part 2.4>

        x = x.unsqueeze(0)

        embedding = self.embed(x) 

        output, hidden_state = self.gru(embedding, prev_hidden_state)

        normalized = self.layer_norm(output)

        logits = self.fc(normalized)

        logits = logits.squeeze(0)

        return logits, hidden_state

# Checkpoint 2.3
def trainLM(model, data, pad_token_id, learning_rate, device):

    #input: model - instance of RecurrentLM to be trained

    #       data - contains X and y as defined in 2.1

    #       pad_token_id - tokenizer’s pad token ID for filtering out pad tokens

    #       learning_rate

    #       device - whether to train model on CPU (="cpu") or GPU (="cuda")

    #output: losses - a list of loss values on the train data from each epoch

    model = model.to(device)

    # ignore the pad tokens by ID and then continue with the training

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # use mini batch gradient descent with Adam optimizer

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # train for 15 epochs

    losses = []

    for epoch in range(15):
        model.train()
        epochLoss = 0.0
        batches = 0

        for Xbatch, ybatch in data:
            Xbatch, ybatch = Xbatch.to(device), ybatch.to(device)

            # Forward pass
            logits, _ = model(Xbatch)

            # convert logits and labels(ybatch) to (#examples, #classes) to use nn.crossentropyloss
            _, _, vocab_size = logits.shape
            logits = logits.reshape(-1, vocab_size)
            ybatch = ybatch.reshape(-1)

            # CrossEntropyloSs
            loss = criterion(logits, ybatch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()
            batches += 1

        avgLoss = epochLoss/batches
        losses.append(avgLoss)

    return losses

# Checkpoint 2.4
def generate(model, tokenizer, start_phrase, max_len, device):

    #input: model - trained instance of RecurrentLM

    #       tokenizer

    #       start_phrase - string containing input word(s)

    #       max_len - max number of tokens to generate

    #       device - whether to inference model on CPU (="cpu") or GPU (="cuda")

    #output: generated_tokens - list of generated token IDs
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize the start phrase
    start_tokens = tokenizer.encode(start_phrase)
    # Convert to tensor and move to device
    input_tokens = torch.tensor([start_tokens]).to(device)
    
    # Use forward method to process initial tokens and get initial hidden state
    with torch.no_grad():
        logits, hidden_state = model(input_tokens)
    
    # Initialize generated tokens list with highest probability logit as next token
    generated_tokens = [start_tokens[-1]]
    
    # Get the last token from the input
    current_token = torch.tensor([start_tokens[-1]]).to(device)
    
    # Generate tokens until max_len or EOS/pad token
    while len(generated_tokens) < max_len:
        # Use stepwise_forward to generate next token
        with torch.no_grad():
            # Get logits for the current token
            next_logits, hidden_state = model.stepwise_forward(current_token, hidden_state)
            
            # Pick token with highest probability (argmax)
            next_token = torch.argmax(next_logits, dim=-1).item()
        
        # Append the token to generated tokens
        generated_tokens.append(next_token)
        
        # Update current token for next iteration
        current_token = torch.tensor([next_token]).to(device)
        
        # Stop if EOS or pad token is generated
        if (next_token == tokenizer.eos_token_id or 
            next_token == tokenizer.pad_token_id):
            break
    
    return generated_tokens

# From Part 1 Checkpoint 1.3
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
        print("Usage: python a2_p2_mahmood_113472709.py <filename>")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    print("\nCheckpoint 2.2:")
    print(f"logits shape: (batch_size, seq_len, vocab_size)")
    print(f"hidden_state shape: (1, batch_size, rnn_hidden_dim)")
    
    # Initialzie RecurrentLM
    vocab_size = GPT2TokenizerFast.get_vocab(gpt2Tokenizer).keys()
    embed_dim = 64
    rnn_hidden_dim = 1024
    model = RecurrentLM(vocab_size=len(vocab_size), embed_dim=embed_dim, rnn_hidden_dim=rnn_hidden_dim)

    # Train LM
    losses = trainLM(model, dataloader,gpt2Tokenizer.pad_token_id, 0.0007, device)

    # Plot loss curves
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title('Training Set Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_curve.pdf')
    plt.close()

    # Use get_perplexity from checkpoint 1.3 to calculate perplexity of the model on below samples

    samples = ["And you gotta live with the bad blood now",
    "Sit quiet by my side in the shade",
    "And I'm not even sorry, nights are so starry",
    "You make me crazier, crazier, crazier, oh",
    "When time stood still and I had you"]

    # Checkpoint 2.3
    print("\nCheckpoint 2.3:")

    model.eval()
    for sample in samples:
        # Tokenize text
        tokens = gpt2Tokenizer.encode(sample)

        # Prep input and target tensors
        input_tensor = torch.tensor(tokens[:-1]).to(device)
        target_tensor = torch.tensor(tokens[1:]).to(device)

        # Get model preds
        with torch.no_grad():
            logits, _ = model(input_tensor)

        # Get probs
        probs = torch.softmax(logits, dim=-1)

        target_probs = []
        for i in range(target_tensor.shape[0]):
            # Get the probability of the target token at each position
            target_probs.append(probs[i, target_tensor[i]].item())
        # Print perplexity for the sample
        perplexity = get_perplexity(target_probs)
        print(f"Sample: \"{sample}\"")
        print(f"Perplexity: {perplexity:.4f}\n")
    print("\nOBSERVATIONS:\nThe model here does indeed perform better than the TrigramLM from Part1, in terms of perplexity and different orders of magnitude depending on the sample. I believe it works better due to a much more thoroughly trained model than just a basic TrigramLM(which essentially just stores counts and guesses). The RNN LM has multiple separate stages, is trained multiple separate times, and optimizes based on loss(with batch gradient descent as we used here). This allows it to be much more accurate than before and as a result have a lower perplexity with the same samples(bar the G/space symbol of course).")
    
    # Checkpoint 2.4
    print("\nCheckpoint 2.4:")

    startPhrases = [
        "<s>Are we",
        "<s>Like we're made of starlight, starlight",
        "<s>Why can't I"  
    ]

    for phrase in startPhrases:
        generated_tokens = generate(model, gpt2Tokenizer, phrase, 64, device)
        
        # Decode the generated tokens to string
        generated_text = gpt2Tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Print the generated text
        print(f"Start phrase \"{phrase}\":\n{generated_text}\n")