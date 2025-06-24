import random, os, sys, math, csv, re, collections, string

import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn, Tensor

import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


import heapq
import matplotlib

# NOTE: COPIED OVER FROM a1_p1 bc couldnt import the function from a1_p1
# Checkpoint 1.1
def wordTokenizer(sent):

    #input: a single sentence as a string.

    #output: a list of each “word” in the text

    # must use regular expressions
    tokens = [] 

    # Check to retain abbreviations of capital letters e.g U.S.A.
    abbrevs = re.findall(r'([A-Z].)+', sent)
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

# Checkpoint 2.0
def getConllTags(filename):

    #input: filename for a conll style parts of speech tagged file

    #output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]

    wordTagsPerSent = [[]]

    sentNum = 0

    with open(filename, encoding='utf8') as f:

        for wordtag in f:

            wordtag=wordtag.strip()

            if wordtag:#still reading current sentence

                (word, tag) = wordtag.split("\t")

                wordTagsPerSent[sentNum].append((word,tag))

            else:#new sentence

                wordTagsPerSent.append([])

                sentNum+=1

    return wordTagsPerSent

# Checkpoint 2.1
def getFeaturesForTarget(tokens, targetI, wordToIndex):
    #input: tokens: a list of tokens in a sentence,

    #       targetI: index for the target token

    #       wordToIndex: dict mapping ‘word’ to an index in the feature list.

    #output: list (or np.array) of k feature values for the given target

    # Extract target index
    targetToken = tokens[targetI]
    
    # 1: First letter capitalized
    firstCapital = 1 if targetToken and targetToken[0].isupper() else 0
    
    # 2: First letter of word
    firstLetterOrd = [0] * 257
    if targetToken:
        char = targetToken[0]
        asciiNum = ord(char)
        if asciiNum <= 255:
            firstLetterOrd[asciiNum] = 1
        else:
            firstLetterOrd[256] = 1 
    
    # 3: Normalize length of the target word
    normLength = min(len(targetToken), 10) / 10 if targetToken else 0
    
    # 4: One-hot of previous word
    prevWordOneHot = [0] * len(wordToIndex)
    if targetI > 0:
        prevWord = tokens[targetI - 1]
        if prevWord in wordToIndex:
            prevWordOneHot[wordToIndex[prevWord]] = 1
    
    # 5: One-hot of current word
    currWordOneHot = [0] * len(wordToIndex)
    if targetToken in wordToIndex:
        currWordOneHot[wordToIndex[targetToken]] = 1
    
    # 6: One-hot of next word
    nextWordOneHot = [0] * len(wordToIndex)
    if targetI < len(tokens) - 1:
        nextWord = tokens[targetI + 1]
        if nextWord in wordToIndex:
            nextWordOneHot[wordToIndex[nextWord]] = 1
    
    # Concatenate all features into one vector
    feature_vector = [firstCapital, normLength] + firstLetterOrd + prevWordOneHot + currWordOneHot + nextWordOneHot
    
    return feature_vector

# Checkpoint 2.2 helper
class LogisticRegressionModel(nn.Module):
    def __init__(self, inSize, outSize):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(inSize, outSize)
    
    def forward(self, x):
        return self.linear(x)

# Checkpoint 2.2
def trainLogReg(train_data, dev_data, learning_rate, l2_penalty):
     #input: train/dev_data - contain the features and labels for train/dev splits
    #input: learning_rate, l2_penalty - hyperparameters for model training
   #output: model - the trained pytorch model
    #output: train/dev_losses - a list of train/dev set loss values from each epoch
    #output: train/dev_accuracies - a list of train/dev set accuracy from each epoch

    xTrain, yTrain = train_data
    xDev, yDev = dev_data
    
    xTrainTensor = torch.tensor(xTrain, dtype=torch.float32)
    yTrainTensor = torch.tensor(yTrain, dtype=torch.long)
    xDevTensor = torch.tensor(xDev, dtype=torch.float32)
    yDevTensor = torch.tensor(yDev, dtype=torch.long)
    
    # Initialize model
    inSize = xTrain.shape[1]
    outSize = len(np.unique(yTrain))
    model = LogisticRegressionModel(inSize, outSize)
   
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)

    criterion = nn.CrossEntropyLoss()
    
    # Lists to store metrics
    trainLosses = []
    trainAccus = []
    devLosses = []
    devAccus = []
    
    # Training loop
    for epoch in range(100):
        model.train()
        
        # Forward pass
        outputs = model(xTrainTensor)
        loss = criterion(outputs, yTrainTensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute training metrics
        _, predicted = torch.max(outputs.data, 1)
        trainAccu = (predicted == yTrainTensor).sum().item() / len(yTrainTensor)
        trainLosses.append(loss.item())
        trainAccus.append(trainAccu)
        
        # Check against dev set
        model.eval()
        with torch.no_grad():
            devOutputs = model(xDevTensor)
            devLoss = criterion(devOutputs, yDevTensor)
            _, devPred = torch.max(devOutputs.data, 1)
            devAccu = (devPred == yDevTensor).sum().item() / len(yDevTensor)
            devLosses.append(devLoss.item())
            devAccus.append(devAccu)
    
    return model, trainLosses, trainAccus, devLosses, devAccus

# Checkpoint 2.3
def gridSearch(train_set, dev_set, learning_rates, l2_penalties):

    #input: learning_rates, l2_penalties - each is a list with hyperparameters to try
   #       train_set - the training set of features and outcomes
   #       dev_set - the dev set of features and outcomes
   #output: model_accuracies - dev set accuracy of the trained model on each hyperparam combination

    #       best_lr, best_l2_penalty - learning rate and L2 penalty combination with highest dev set accuracy

    bestAccu = 0
    best_lr = None
    best_l2_penalty = None
    model_accuracies = {}
    
    for lr in learning_rates:
        model_accuracies[lr] = {}
        for l2 in l2_penalties:
            # Retrain model
            model, _, _, _, dev_accuracies = trainLogReg(
                train_set, dev_set, learning_rate=lr, l2_penalty=l2
            )
            finalAccu = dev_accuracies[-1]
            model_accuracies[lr][l2] = finalAccu
            
            if finalAccu > bestAccu:
                best_lr = lr
                best_l2_penalty = l2
                bestAccu = finalAccu
               

    return model_accuracies, best_lr, best_l2_penalty

# Checkpoint 2.4
def predictTags(model_params, tokens, wordToIndex, tagToIndex):
    featureVecs = np.array([getFeaturesForTarget(tokens, i, wordToIndex) for i in range(len(tokens))])
    
    # Convert to tensor
    featureTensor = torch.tensor(featureVecs, dtype=torch.float32)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(featureTensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # Convert indices to tags
    tags = {v: k for k, v in tagToIndex.items()}
    pred = [tags[i.item()] for i in predicted]
    
    return pred

if __name__=="__main__":

    # Checkpoint 2.0
    # Load pos data

    posPairs = getConllTags("daily547_3pos.txt")

    # Creating dictionary of all unique tokens and tags

    tokens = []
    tags = []
    
    for pair in posPairs:
        for word, tag in pair:
            tokens.append(word)
            tags.append(tag)
    
    # Create dictionaries
    uniqueTokens = {word: i for i, word in enumerate(set(tokens))}
    uniqueTags = {tag: i for i, tag in enumerate(set(tags))}

    # Checkpoint 2.1
        
    # Create feature matrix and ground truth vector
    X = []
    y = []
    
    for pair in posPairs:
        tokens = [word for word, _ in pair]
        for i, (word, tag) in enumerate(pair):
            X.append(getFeaturesForTarget(tokens, i, uniqueTokens))
            y.append(uniqueTags[tag])

    X= np.array(X)
    y= np.array(y)
    
    # Split X and y into train and dev subsets
    xTrain, xDev, yTrain, yDev = train_test_split(X, y, train_size=0.7, random_state=39)

    print("Checkpoint 2.1:")
    vectorSum = np.sum(X[:5], axis=0) + np.sum(X[-5:], axis=0)
    print(f"Sum of first & last 5 feature vectors of X: {vectorSum}")
    
    # Checkpoint 2.2
    # Train logistic regression model
    trainData = (xTrain, yTrain)
    devData = (xDev, yDev)
    
    model, trainLosses, trainAccus, devLosses, devAccus = trainLogReg(trainData, devData, 0.01, 0.01)

    print("\nCheckpoint 2.2:")
    
    # Plot training and dev loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label='Training Loss')
    plt.plot(devLosses, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Dev Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(trainAccus, label='Training Accuracy')
    plt.plot(devAccus, label='Dev Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Dev. Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_accuracy_curves.pdf')


    # Checkpoint 2.3
    learningRates = [0.1, 1, 10]
    L2Penalties = [1e-5, 1e-3, 1e-1]

    model_accuracies, best_lr, best_l2_penalty = gridSearch(trainData, devData, learningRates, L2Penalties)

    print("\nCheckpoint 2.3:")
    print("Dev set accuracy for different hyperparameters:")
    
    print("LR\\L2\t\t1e-5\t\t1e-3\t\t1e-1")
    for lr in learningRates:
        print(f"{lr}\t\t", end="")
        for l2 in L2Penalties:
            print(f"{model_accuracies[lr][l2]:.4f}\t\t", end="")
        print()
    
    print(f"\nBest hyperparameters: Learning Rate = {best_lr}, L2 Penalty = {best_l2_penalty}")
    
    # Get best model loss and accu curves
    best_model, best_train_losses, best_train_accuracies, best_dev_losses, best_dev_accuracies = trainLogReg(trainData, devData, best_lr, best_l2_penalty)
    
    # Plot best model curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(best_train_losses, label='Training Loss')
    plt.plot(best_dev_losses, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Best Model Loss Curves (LR={best_lr}, L2={best_l2_penalty})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(best_train_accuracies, label='Training Accuracy')
    plt.plot(best_dev_accuracies, label='Dev Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Best Model Accuracy Curves (LR={best_lr}, L2={best_l2_penalty})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('best_model_curves.pdf')

    # Checkpoint 2.4
    sampleSentences = ['The horse raced past the barn fell.', 'For 3 years, we attended S.B.U. in the CS program.',
                       'Did you hear Sam tell me to "chill out" yesterday? #rude']
    for sentence in sampleSentences:
        tokens = wordTokenizer(sentence)
        predTags = predictTags(best_model, tokens, uniqueTokens, uniqueTags)
        
        print(f"\nSample sentence : {sentence}")
        print("Token\t\tPredicted POS")
        print("-" * 30)
        for token, tag in zip(tokens, predTags):
            print(f"{token:<15}\t{tag}")