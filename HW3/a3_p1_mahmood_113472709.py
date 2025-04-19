import numpy as np
import torch

import sklearn.metrics
import tqdm

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup

# Parts I and II
boolq_dataset = load_dataset('google/boolq')

# ------------Helper functions------------------
def printMetrics(preds, trueLabels):
    accuracy = sklearn.metrics.accuracy_score(trueLabels, preds)
    f1 = sklearn.metrics.f1_score(trueLabels, preds, average='macro')

    # Class specific metruics
    classPrec = sklearn.metrics.precision_score(trueLabels, preds, average=None)
    classRec = sklearn.metrics.recall_score(trueLabels, preds, average=None)
    classF1 = sklearn.metrics.f1_score(trueLabels, preds, average=None)

    # Print metrics
    print(f"Overall: acc: {accuracy:.3f}, f1: {f1:.3f}")
    print(f"    Yes: prec: {classPrec[1]:.3f}, rec: {classRec[1]:.3f}, f1: {classF1[1]:.3f}")
    print(f"    No:  prec: {classPrec[0]:.3f}, rec: {classRec[0]:.3f}, f1: {classF1[0]:.3f}")
    return

def plotLosses(losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def prepGPT2Input(item, tokenizer, max_length=1024):
    passage = item['passage']
    question = item['question']
    answer = "yes" if item['answer'] else "no"
    
    # Format input
    text = f"{passage}\n{question}?\n"
    fullInput = text + answer

    inputs = tokenizer(fullInput, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    return inputs

def initGPT2Loader(dataset, tokenizer, batch_size=8, shuffle=True):
    processed = []
    
    for i in range(len(dataset)):
        processed.append(prepGPT2Input(dataset[i], tokenizer))
    
    # Pad per max seq length in batch instead of max seq length in dataset everytime - TA
    def collate_fn(batch):
        batchLen = max(item["input_ids"].size(0) for item in batch)
        
        collated = {}
        for key in batch[0].keys():
            if key in ["input_ids", "attention_mask", "labels"]:
                paddedTensors = []
                for item in batch:
                    tensor = item[key]
                    currLen = tensor.size(0)
                    if currLen < batchLen:
                        if key == "labels":
                            padVal = -100
                        elif key == "attention_mask":
                            padVal = 0
                        else: 
                            padVal = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                        
                        padded = torch.cat([tensor,torch.full((batchLen - currLen,), padVal, dtype=tensor.dtype)])
                        paddedTensors.append(padded)
                    else:
                        paddedTensors.append(tensor)
                
                collated[key] = torch.stack(paddedTensors)
        
        return collated
    
    return DataLoader(processed, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def prepRobertaInput(item, tokenizer, max_length=512):
    passage = item['passage']
    question = item['question']
    answer = 1 if item['answer'] else 0
    
    # Format input
    text = f"{passage}\n{question}?\n"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    return inputs, answer

def initRobertaLoader(dataset, tokenizer, batch_size=16, shuffle=True):
    inputIDs = []
    attMasks = []
    labels = []
    
    for i in range(len(dataset)):
        inputs, answer = prepRobertaInput(dataset[i], tokenizer)
        inputIDs.append(inputs["input_ids"])
        attMasks.append(inputs["attention_mask"])
        labels.append(answer)
    
    # Convert - tensors
    inputIDs = torch.stack(inputIDs)
    attMasks = torch.stack(attMasks)
    labels = torch.tensor(labels)
    
    tensor_dataset = torch.utils.data.TensorDataset(inputIDs, attMasks, labels)
    
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


# ------------Part 1.1------------------

def evalGPT2(model, dataset, tokenizer, device):
    model.eval()
    model.to(device)

    preds = []
    trueLabels = []

    for i in range(len(dataset)):
        item = dataset[i]
        trueLabels.append(item['answer'])
        passage = item['passage']
        question = item['question']

        # Format input and tokenize
        input = f"{passage}\n{question}?\n"
        inputIds = tokenizer(input, return_tensors='pt', truncation=True, max_length=1024).to(device)

        # Encode and get logits
        yesID = tokenizer.encode("yes")[0]
        noID = tokenizer.encode("no")[0]

        with torch.no_grad():
            outputs = model(**inputIds)
            logits = outputs.logits
        
        # Get logits for last token
        lastTokenLogits = logits[0, -1, :]

        # Get probs
        yesProb = torch.softmax(lastTokenLogits, dim=0)[yesID].item()
        noProb = torch.softmax(lastTokenLogits, dim=0)[noID].item()

        # Get pred
        pred = yesProb > noProb
        preds.append(pred)

    return preds, trueLabels

# ------------Part 1.2------------------

def finetuneGPT2(model, train_loader, num_epochs=1, lr=1e-5, weight_decay=1e-3, device='cuda'):
    model.to(device)
    model.train()
    
    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Get totasl steps
    totalSteps = len(train_loader) * num_epochs
    
    # LR sched
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=totalSteps)
    
    batchLosses = []
    
    for _ in range(num_epochs):        
        for batch in tqdm.tqdm(train_loader):
            # Move batch 
            inputIDs = batch["input_ids"].to(device)
            attMask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            # Use this to do it faster and alllow bigger batches - TA
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
              outputs = model(input_ids=inputIDs,attention_mask=attMask,labels=labels)
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batchLosses.append(loss.item())
            
    return model, batchLosses

# ------------Part 1.4------------------

def finetuneRoberta(model, train_loader, num_epochs=1, lr=1e-5, weight_decay=1e-3, device='cuda'):
    model.to(device)
    model.train()
    
    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Get total steps
    totalSteps = len(train_loader) * num_epochs

    # LR sched
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=totalSteps)
    
    batchLosses = []
    
    
    for _ in range(num_epochs):
        for inputIDs, attMask, labels in tqdm.tqdm(train_loader):
            # Move batch to device
            inputIDs = inputIDs.to(device)
            attMask = attMask.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=inputIDs, attention_mask=attMask)
                logits = outputs.logits

                # Change to binary cross entropy loss
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(logits[:, 1], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batchLosses.append(loss.item())
            
    return model, batchLosses

def evalRoberta(model, dataset, device='cuda'):
    model.eval()
    model.to(device)
    
    preds = []
    trueLabels = []
    
    with torch.no_grad():
        for inputIDs, attMask, labels in dataset:
            # Move batch to device
            inputIDs = inputIDs.to(device)
            attMask = attMask.to(device)
            
            # Forward pass
            outputs = model(input_ids=inputIDs,attention_mask=attMask)

            # Get preds
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            
            preds.extend(pred)
            trueLabels.extend(labels.numpy())
    
    return preds, trueLabels

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Checkpoint 1.1
    print("Checkpoint 1.1")

    gpt2Tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    gpt2Model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token

    preds, trueLabels = evalGPT2(gpt2Model, boolq_dataset['validation'], gpt2Tokenizer, device)

    # Print metrics
    printMetrics(preds, trueLabels)

    # Checkpoint 1.2
    print("Checkpoint 1.2")
    torch.cuda.empty_cache()

    # Create train dataloader for GPT2
    trainLoader = initGPT2Loader(boolq_dataset['train'], gpt2Tokenizer)
    
    # Fine tune
    gpt2ModelFinetuned, gpt2Losses = finetuneGPT2(gpt2Model, trainLoader)

    # Plot loss curve
    plotLosses(gpt2Losses, "GPT2 Loss Curve", "gpt2_loss_curve.png")
  
    # Checkpoint 1.3
    print("Checkpoint 1.3")
    torch.cuda.empty_cache()
    
    # Eval model
    preds, trueLabels = evalGPT2(gpt2ModelFinetuned, boolq_dataset['validation'], gpt2Tokenizer, device)
    
    # Print metrics
    printMetrics(preds, trueLabels)

    # Checkpoint 1.4
    print("Checkpoint 1.4")
    torch.cuda.empty_cache()

    # Load model and tokenizer
    robertaTokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
    robertaModel = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
    
    # Init dataloaders
    robertaTrain = initRobertaLoader(boolq_dataset['train'], robertaTokenizer)
    robertaVal = initRobertaLoader(boolq_dataset['validation'], robertaTokenizer, batch_size=16, shuffle=False)
    
    # Fine tune RoBERTa
    robertaModelFinetuned, robertaLosses = finetuneRoberta(robertaModel,robertaTrain)
    
    # Plot loss curve
    plotLosses(robertaLosses, "RoBERTa Loss Curve", "roberta_loss_curve.png")

    # Eval model
    preds, trueLabels = evalRoberta(robertaModelFinetuned, robertaVal, device)
    
    # Print metrics
    printMetrics(preds,trueLabels)
    torch.cuda.empty_cache()

