import random, os, sys, math, csv, re, collections, string

import numpy as np

import torch

from torch import nn, Tensor

import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


import heapq
import matplotlib

# Checkpoint 1.1
def wordTokenizer(sent):

    #input: a single sentence as a string.

    #output: a list of each “word” in the text

    # must use regular expressions

    tokens = [] 

    # Check to retain abbreviations of capital letters e.g U.S.A.
    abbrevs = re.findall(r'([A-Z]\.)+(([A-Z]\.)+)?', sent)
    for i, abbr in enumerate(abbrevs):
        sent = sent.replace(abbr, f"ABBR{i}",1)

    # Check to retain periods surrounded by integers e.g 6.0    
    nums = re.findall(r'(\d+\.\d+)', sent)
    for i, num in enumerate(nums):
        sent = sent.replace(num, f"NUM{i}",1)

    # Check to retain contractions e.g. don't, can't, won't, etc.
    contractions = re.findall(r"(\w+'\w+)", sent)
    for i, con in enumerate(contractions):
        sent = sent.replace(con, f"CON{i}",1)

    # Check to retain hashtags and @mentions e.g. #bestie, @bestie
    hashtags = re.findall(r'(#\w+)', sent)
    for i, tag in enumerate(hashtags):
        sent = sent.replace(tag, f"TAG{i}",1)

    mentions = re.findall(r'(@\w+)', sent)
    for i, ment in enumerate(mentions):
        sent = sent.replace(ment, f"MENT{i}",1)

    # Separate all punctuation from words
    sent = re.sub(r'([^\w\s])', r' \1 ', sent)

    # Replace all spaces with a single space
    sent = re.sub(r'\s+', ' ', sent).strip()

    # Return all placeholders to their original form
    for i, abbr in enumerate(abbrevs):
        sent = sent.replace(f"ABBR{i}", abbr)
    for i, num in enumerate(nums):
        sent = sent.replace(f"NUM{i}", num)
    for i, con in enumerate(contractions):
        sent = sent.replace(f"CON{i}", con)
    for i, tag in enumerate(hashtags):
        sent = sent.replace(f"TAG{i}", tag)
    for i, ment in enumerate(mentions):
        sent = sent.replace(f"MENT{i}", ment)

    return sent.split()

# Checkpoint 1.2
# def spacelessBPELearn(docs, max_vocabulary=1000):


#    return final_vocabulary

# def spacelessBPETokenize(text, vocab):

#     #input: text, a single string to be word tokenized.

#     #       vocab, a set of valid vocabulary words

#     #output: words, a list of strings of all word tokens, in order, from the string

#    return words

def spacelessBPELearn(docs, max_vocabulary=1000):
    
    #input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary

    #output: final_vocabulary, a set of all members of the learned vocabulary

    # Start with vocabulary of all ascii letters as words
    vocab = set(list("0123456789"))  # 0-9
    vocab.update(chr(i) for i in range(97, 123))  # a-z
    vocab.update(chr(i) for i in range(65, 91))  # A-Z
    
    # Process the corpus
    words = []
    for doc in docs:
        # Replace non-ASCII characters with '?'
        doc = ''.join(c if c.isascii() else '?' for c in doc)
        # Split by whitespace
        doc_words = doc.split()
        words.extend(doc_words)
    
    # Convert each word to a list of characters
    word_tokens = [[c if c.isascii() else '?' for c in word] for word in words]
    
    # Start the BPE algorithm
    while len(vocab) < max_vocabulary:
        # Find the most frequent pair
        pairs = collections.defaultdict(int)
        for word in word_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] += 1
        
        if not pairs:
            break
        
        # Get the most frequent pair
        best_pair = max(pairs.items(), key=lambda x: x[1])[0]
        
        # Create a new token by combining the pair
        new_token = ''.join(best_pair)
        vocab.add(new_token)
        
        # Replace all occurrences of the pair with the new token
        new_word_tokens = []
        for word in word_tokens:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tokens.append(new_word)
        
        word_tokens = new_word_tokens
        
        # Print iterations at specified points
        vocab_size = len(vocab)
        if vocab_size in [0+26+26+10, 1+26+26+10, 10+26+26+10, 100+26+26+10, 500+26+26+10]:
            print(f"Iteration {vocab_size-26-26-10}: Most frequent pair: {best_pair} -> {new_token} (count: {pairs[best_pair]})")
    
    # Flatten the list of tokens in each word and get unique tokens
    final_vocabulary = vocab
    
    return final_vocabulary

def spacelessBPETokenize(text: str, vocab):
    """
    Tokenizes text using a spaceless BPE tokenizer.
    
    Args:
        text: A single string to be word tokenized
        vocab: A set of valid vocabulary words
    
    Returns:
        A list of strings of all word tokens, in order, from the string
    """
    # Replace non-ASCII characters with '?'
    text = ''.join(c if c.isascii() else '?' for c in text)
    
    # Split by whitespace to get words
    raw_words = text.split()
    result = []
    
    # Process each word
    for word in raw_words:
        # Initialize with characters
        chars = list(word)
        
        # Apply BPE merges until no more can be applied
        while True:
            # Find all pairs
            pairs = [(chars[i], chars[i+1]) for i in range(len(chars) - 1)]
            
            # Find valid merges based on vocabulary
            valid_merges = [(i, ''.join(pairs[i])) for i in range(len(pairs)) if ''.join(pairs[i]) in vocab]
            
            if not valid_merges:
                break
            
            # Apply the first valid merge (leftmost)
            pos, merged = valid_merges[0]
            new_chars = chars[:pos] + [merged] + chars[pos+2:]
            chars = new_chars
        
        # Add the tokenized word to the result
        result.extend(chars)
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python a1_p1_mahmood_113472709.py <train_file>")
        sys.exit(1)
    
    # Grab input file and read in the data
    file = sys.argv[1]
    docs = []
    with open(file, 'r', encoding='utf-8') as f:
        docs = [line.strip() for line in f.readlines()]
    
    # Checkpoint 1.1 
    # Print first 5 and last doc
    print("Checkpoint 1.1:")
    for i in range(0,5):
        print(wordTokenizer(docs[i]))
    print(wordTokenizer(docs[-1]))


    # Checkpoint 1.2 
    # Print top 5 most frequent pairs at iterations 0,1, 10, 100 and 500

    # Print final vocabulary

    # Print first 5 and last doc
    # print("Checkpoint 1.2:")
    # for i in range(0,5):
    #     print(wordTokenizer(docs[i]))
    # print(wordTokenizer(docs[-1]))

       # Checkpoint 1.2
    print("\nCheckpoint 1.2:")
    print("Learning BPE vocabulary...")
    vocabulary = spacelessBPELearn(docs)
    
    print(f"\nFinal vocabulary size: {len(vocabulary)}")
    print(f"Final vocabulary (sample of 10): {list(vocabulary)[:10]}...")
    
    print("\nTokenization of documents using BPE:")
    for i in range(5):
        print(f"Document {i+1}: {docs[i]}")
        print(f"BPE Tokens: {spacelessBPETokenize(docs[i], vocabulary)}")
    
    print(f"Document {len(docs)}: {docs[-1]}")
    print(f"BPE Tokens: {spacelessBPETokenize(docs[-1], vocabulary)}")

    pass