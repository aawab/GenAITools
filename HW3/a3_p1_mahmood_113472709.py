import os, sys, random, re, collections, string
from matplotlib.pylab import rec
import numpy as np
import torch

import math
import csv

import sklearn.model_selection
import sklearn.metrics

import heapq
import matplotlib
import tqdm

import transformers
import datasets

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Parts I and II

boolq_dataset = load_dataset('google/boolq')

# You may now use any AutoModel for getting the LMs. You are welcome to reference any examples from Hugging Face themselves.

# ------------Helper functions------------------
# For part 1.1-1.4 metric calcs
def printMetrics(preds, trueLabels):
    accuracy = sklearn.metrics.accuracy_score(trueLabels, preds)
    f1_macro = sklearn.metrics.f1_score(trueLabels, preds, average='macro')

    # Class-specific precision, recall, f1-score
    class_precisions = sklearn.metrics.precision_score(trueLabels, preds, average=None)
    class_recalls = sklearn.metrics.recall_score(trueLabels, preds, average=None)
    class_f1s = sklearn.metrics.f1_score(trueLabels, preds, average=None)

    # Print metrics
    print(f"Overall: acc: {accuracy:.3f}, f1: {f1_macro:.3f}")
    print(f"    Yes: prec: {class_precisions[1]:.3f}, rec: {class_recalls[1]:.3f}, f1: {class_f1s[1]:.3f}")
    print(f"    No:  prec: {class_precisions[0]:.3f}, rec: {class_recalls[0]:.3f}, f1: {class_f1s[0]:.3f}")
    return

################  Part 1.1  ################

def evalGPT2(model, dataset, tokenizer, device):
    """
    Function to implement zero-shot distilGPT2 inference
    Args:
    model: GPT2 model
    dataset: Dataset directly indexing for 'validation'
    ...: Any other arguments you may need

    Returns:
    preds and true labels for printmetric
    """
    model.eval()
    model.to(device)

    preds = []
    trueLabels = []

    for i in range(len(dataset)):
        item = dataset[i]
        question = item['question']
        passage = item['passage']
        trueLabels.append(item['answer'])

        # Format input and tokenize
        input = f"{passage}\n{question}?\n"
        inputIds = tokenizer(input, return_tensors='pt', truncation=True, max_length=1024).to(device)

        # Encode and get logits
        yesID = tokenizer.encode(" yes")[0]
        noID = tokenizer.encode(" no")[0]

        if i==0:
            print(f"Yes token id: {yesID}, token: {tokenizer.decode([yesID])}")
            print(f"No token id: {noID}, token: {tokenizer.decode([noID])}")

        with torch.no_grad():
            outputs = model(**inputIds)
            logits = outputs.logits
        
        # Get logits for last token
        lastTokenLogits = logits[0, -1, :]

        # Get probabilities and 
        yesProb = torch.softmax(lastTokenLogits, dim=0)[yesID].item()
        noProb = torch.softmax(lastTokenLogits, dim=0)[noID].item()

        # Get prediction
        pred = yesProb > noProb
        preds.append(pred)
    # Print truelabel shape and preds shape
    print(f"True labels shape: {len(trueLabels)}")
    print(f"Predictions shape: {len(preds)}")
    return preds, trueLabels

################  Part 1.2  ################

def finetune_gpt2_classifier(model, train_loader, num_epochs=2, lr=1e-5, weight_decay=1e-3, device,...):
    """
    Function to fine-tune a distilGPT2 model for a classification task
    Args:
    model: instance of distilGPT2
    train_loader: Dataloader for the BoolQ training set
    num_epochs: Number of epochs for training
    lr: Learning rate
    weight_decay: Weight decay
    ...: Any other arguments you may need

    Returns:
    model: Fine-tuned model
    batch_losses: List of losses for each mini-batch
    """

    model.to(device)
    model.train()

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    batch_losses = []

    # Calc total training steps
    total_steps = len(train_loader) * num_epochs

    # Learning rate scheduler
    

    return batch_losses


################  Part 1.4  ################
#
# def finetune_roberta_classifier(model, train_loader, num_epochs, lr, weight_decay, ...):
#     """
#     Function to fine-tune a distilRoberta model for a classification task
#     Args:
#     model: instance of distilRoberta
#     train_loader: Dataloader for the BoolQ training set
#     num_epochs: Number of epochs for training
#     lr: Learning rate
#     weight_decay: Weight decay
#     ...: Any other arguments you may need
# 
#     Returns:
#     model: Fine-tuned model
#     batch_losses: List of losses for each mini-batch
#     """
#     #<FILL IN>
#     return batch_losses
#
#
# def evaluate_roberta_classifier(model, data_loader, ...):
#     """
#     Function to implement distilRoberta inference
#     Args:
#     model: instance of distilRoberta
#     data_loader: Dataloader for the dataset
#     ...: Any other arguments you may need
#
#     Returns:
#     preds
#     """
#     #<FILL IN: code to evaluate the model and print the results>


if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Checkpoint 1.1
    print("Checkpoint 1.1")

    # Load model, tokenizer, dataset
    gpt2tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    gpt2model = GPT2LMHeadModel.from_pretrained('distilgpt2')

    preds, trueLabels = evalGPT2(gpt2model, boolq_dataset['validation'], gpt2tokenizer, device)

    # Get metrics and print them
    printMetrics(preds, trueLabels)
    # TODO: SOMEHOW MAKE IT MORE ACCURATE, RN OVERAL ACC 0.386, nEED >=0.4. overall f1 0.306, need >0.35

    # Checkpoint 1.2


    # Checkpoint 1.3


    # Checkpoint 1.4
    pass