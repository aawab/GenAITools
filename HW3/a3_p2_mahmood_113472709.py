import re
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics
import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification

# Parts I and II
boolq_dataset = load_dataset('google/boolq')

# Part II
sst_dataset = load_dataset('stanfordnlp/sst', trust_remote_code=True)

# ------------Helper functions------------------

def printMetrics(preds, trueLabels):
    f1 = sklearn.metrics.f1_score(trueLabels, preds, average='macro')
    accuracy = sklearn.metrics.accuracy_score(trueLabels, preds)
    print(f"overall acc: {accuracy:.3f}, f1: {f1:.3f}")
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

def prepRobertaInput(item, tokenizer, max_length=512):
    passage = item['passage']
    question = item['question']
    answer = 1 if item['answer'] else 0
    
    # Format input
    text = f"{passage}\n{question}?\n"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}  
    return inputs, answer

def initRobertaLoader(dataset, tokenizer, batch_size=16, shuffle=True):
    processed = []

    for i in range(len(dataset)):
        inputs, answer = prepRobertaInput(dataset[i], tokenizer)
        processed.append((inputs["input_ids"], inputs["attention_mask"], answer))
    
    def collate(batch):
        inputIDs = [item[0] for item in batch]
        attMasks = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        # Find max length for batch
        batchLen = max(len(ids) for ids in inputIDs)
        
        paddedInputs = []
        paddedAttMasks = []
        
        for ids, mask in zip(inputIDs, attMasks):
            padLen = batchLen - len(ids)
            
            paddedIDs = torch.cat([ids, torch.full((padLen,), tokenizer.pad_token_id, dtype=torch.long)])
            
            paddedMask = torch.cat([mask,torch.zeros(padLen, dtype=torch.long)])
            
            paddedInputs.append(paddedIDs)
            paddedAttMasks.append(paddedMask)
        
        inputTensor = torch.stack(paddedInputs)
        attMasksTensor = torch.stack(paddedAttMasks)
        labelsTensor = torch.tensor(labels)
        
        return inputTensor, attMasksTensor, labelsTensor
    
    return DataLoader(processed,batch_size=batch_size,shuffle=shuffle,collate_fn=collate)

def prepRobertaRegressionInput(item, tokenizer, max_length=512):
    sentence = item['sentence']
    label = item['label']
    
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    return inputs, label

def initRobertaRegressionLoader(dataset, tokenizer, batch_size=16, shuffle=True):
    processed = []
    
    for i in range(len(dataset)):
        inputs, label = prepRobertaRegressionInput(dataset[i], tokenizer)
        processed.append((inputs["input_ids"], inputs["attention_mask"], label))
    
    def collate(batch):
        inputIDs = [item[0] for item in batch]
        attMasks = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        batchLen = max(len(ids) for ids in inputIDs)
        
        paddedInputs = []
        paddedAttMasks = []
        
        for ids, mask in zip(inputIDs, attMasks):
            padLen = batchLen - len(ids)
            
            paddedIDs = torch.cat([ids,torch.full((padLen,), tokenizer.pad_token_id, dtype=torch.long)])
            paddedMask = torch.cat([mask,torch.zeros(padLen, dtype=torch.long)])
            
            paddedInputs.append(paddedIDs)
            paddedAttMasks.append(paddedMask)
        
        inputTensor = torch.stack(paddedInputs)
        attMasksTensor = torch.stack(paddedAttMasks)
        labelsTensor = torch.tensor(labels, dtype=torch.float)
        
        return inputTensor, attMasksTensor, labelsTensor
    
    return DataLoader(processed,batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

# ------------Part 2.1------------------

def convertRBRand(model):
    def randWeights(module):
        if isinstance(module, nn.Linear):
            # random gaussian for linear weights
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # 1s for norm weights
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    model.apply(randWeights)
    
    return model

def convertRBKQV(model):
    for i in range(model.config.num_hidden_layers - 2, model.config.num_hidden_layers):
        layer = model.roberta.encoder.layer[i]
        
        qWeight = layer.attention.self.query.weight
        kWeight = layer.attention.self.key.weight
        
        qBias = layer.attention.self.query.bias
        kBias = layer.attention.self.key.bias
        
        sharedWeight = (qWeight + kWeight) / 2
        sharedBias = (qBias + kBias) / 2
        
        hiddenSize = model.config.hidden_size
        sharedKQV = nn.Linear(hiddenSize, hiddenSize)
        sharedKQV.weight.data = sharedWeight.clone()
        sharedKQV.bias.data = sharedBias.clone()
        
        layer.attention.self.query = sharedKQV
        layer.attention.self.key = sharedKQV
        layer.attention.self.value = sharedKQV
    
    return model

def convertRBNores(model):
    for i in range(model.config.num_hidden_layers - 2, model.config.num_hidden_layers):
        layer = model.roberta.encoder.layer[i]
        
        def newAttOutputForward(self, hiddenStates, inputTensor):
            hiddenStates = self.dense(hiddenStates)
            hiddenStates = self.dropout(hiddenStates)
            hiddenStates = self.LayerNorm(hiddenStates)
            return hiddenStates
            
        def newOutputForward(self, hiddenStates, inputTensor):
            hiddenStates = self.dense(hiddenStates)
            hiddenStates = self.dropout(hiddenStates)
            hiddenStates = self.LayerNorm(hiddenStates)
            return hiddenStates
            
        def newLayerForward(self, hiddenStates, attentionMask=None, headMask=None, *args, **kwargs):
            selfOutputs = self.attention.self(hiddenStates, attentionMask, headMask, *args, **kwargs)
            attentionOutput = newAttOutputForward(self.attention.output, selfOutputs[0], hiddenStates)
            outputs = (attentionOutput,) + selfOutputs[1:]
            
            intermediateOutput = self.intermediate(attentionOutput)
            layerOutput = newOutputForward(self.output, intermediateOutput, attentionOutput)
            
            return (layerOutput,) + outputs[1:]
        
        layer.attention.output.forward = lambda hiddenStates, inputTensor, *args, **kwargs: newAttOutputForward(layer.attention.output, hiddenStates, inputTensor)
        layer.output.forward = lambda hiddenStates, inputTensor, *args, **kwargs: newOutputForward(layer.output, hiddenStates, inputTensor)
        layer.forward = lambda *args, **kwargs: newLayerForward(layer, *args, **kwargs)
        
    return model

def getModel(variant='distilroberta', task='classifier'):
    if task == 'classifier':
        model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
    else:  # regression task
        model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=1,problem_type="regression")
    
    if variant == 'distilRB-rand':
        model = convertRBRand(model)
    elif variant == 'distilRB-KQV':
        model = convertRBKQV(model)
    elif variant == 'distilRB-nores':
        model = convertRBNores(model)
    
    return model

# ------------Part 2.2 and 2.3------------------

def finetuneRoberta(model, train_loader, num_epochs=1, lr=1e-5, weight_decay=1e-3, device='cuda', regression=False):
    model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Get total steps
    totalSteps = len(train_loader) * num_epochs

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=totalSteps)

    batchLosses = []
    
    for _ in range(num_epochs):
        for inputIDs, attMask, labels in tqdm.tqdm(train_loader):
            # Move batch
            inputIDs = inputIDs.to(device)
            attMask = attMask.to(device)
            labels = labels.to(device)
            
            if regression:
                labels = labels.float().view(-1, 1)

            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=inputIDs, attention_mask=attMask, labels=labels)
                loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batchLosses.append(loss.item())
            
    return model, batchLosses

def evalRobertaClassifier(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    preds = []
    trueLabels = []
    
    with torch.no_grad():
        for inputIDs, attMask, labels in test_loader:
            # Move batch to device
            inputIDs = inputIDs.to(device)
            attMask = attMask.to(device)
            
            # Forward pass
            outputs = model(input_ids=inputIDs, attention_mask=attMask)
            
            # Get predictions
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            
            preds.extend(pred)
            trueLabels.extend(labels.numpy())
    
    return preds, trueLabels

def evalRobertaRegressor(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    preds = []
    trueLabels = []
    
    with torch.no_grad():
        for inputIDs, attMask, labels in test_loader:
            # Move batch to device
            inputIDs = inputIDs.to(device)
            attMask = attMask.to(device)
            
            # Forward pass
            outputs = model(input_ids=inputIDs, attention_mask=attMask)
            
            predictions = outputs.logits.squeeze(-1)
            
            preds.extend(predictions.cpu().numpy())
            trueLabels.extend(labels.numpy())
    
    preds = np.array(preds)
    trueLabels = np.array(trueLabels)
    
    mae = np.mean(np.abs(preds - trueLabels))   
    r = np.corrcoef(preds, trueLabels)[0, 1]
    
    return mae, r

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
    
    print("Checkpoint 2.2")
    
    trainLoader = initRobertaLoader(boolq_dataset['train'], tokenizer)
    valLoader = initRobertaLoader(boolq_dataset['validation'], tokenizer, batch_size=16, shuffle=False)
    
    rbRand = getModel(variant='distilRB-rand', task='classifier')
    roberta = getModel(variant='distilroberta', task='classifier')
    rbKQV = getModel(variant='distilRB-KQV', task='classifier')
    rbNores = getModel(variant='distilRB-nores', task='classifier')
    
    # Fine tune base 
    roberta, _ = finetuneRoberta(roberta, trainLoader)
    
    # Fine tune KQV
    rbKQV, _ = finetuneRoberta(rbKQV, trainLoader)
    
    # Fine tune nores
    rbNores, _ = finetuneRoberta(rbNores, trainLoader)
    
    # Evaluate rand
    print("distilRB-rand: ", end="")
    preds, trueLabels = evalRobertaClassifier(rbRand, valLoader, device)
    printMetrics(preds, trueLabels)
    
    # Evaluate roberta
    print("distilroberta: ", end="")
    preds, trueLabels = evalRobertaClassifier(roberta, valLoader, device)
    printMetrics(preds, trueLabels)
    
    # Evaluate KQV
    print("distilRB-KQV: ", end="")
    preds, trueLabels = evalRobertaClassifier(rbKQV, valLoader, device)
    printMetrics(preds, trueLabels)
    
    # Evaluate nores
    print("distilRB-nores: ", end="")
    preds, trueLabels = evalRobertaClassifier(rbNores, valLoader, device)
    printMetrics(preds, trueLabels)
    
    # Checkpoint 2.3
    print("Checkpoint 2.3")
    
    trainLoader = initRobertaRegressionLoader(sst_dataset['train'], tokenizer)
    valLoader = initRobertaRegressionLoader(sst_dataset['validation'], tokenizer, batch_size=16, shuffle=False)
    testLoader = initRobertaRegressionLoader(sst_dataset['test'], tokenizer, batch_size=16, shuffle=False)
    
    rbRandReg = getModel(variant='distilRB-rand', task='regressor')
    robertaReg = getModel(variant='distilroberta', task='regressor')
    rbKQVReg = getModel(variant='distilRB-KQV', task='regressor')
    rbNoresReg = getModel(variant='distilRB-nores', task='regressor')
    
    # Fine tune roberta
    robertaReg, robertaLosses = finetuneRoberta(robertaReg, trainLoader, regression=True)
    
    # Only plot for roberta
    plotLosses(robertaLosses, "DistilRoBERTa Regression Loss Curve", "distilroberta_regression_loss_curve.png")
    
    # Fine tune KQV 
    rbKQVReg, _ = finetuneRoberta(rbKQVReg, trainLoader,regression=True)
    
    # Fine tune nores
    rbNoresReg, _ = finetuneRoberta(rbNoresReg, trainLoader,regression=True)
        
    valMae, valR = evalRobertaRegressor(robertaReg, valLoader, device)
    testMae, testR = evalRobertaRegressor(robertaReg, testLoader, device)
    print(f"Validation: mae: {valMae:.3f}, r: {valR:.3f}")
    print(f"Test: mae: {testMae:.3f}, r: {testR:.3f}")
    
    print("\nSST test set:")
    
    # Eval rand
    testMae, testR = evalRobertaRegressor(rbRandReg, testLoader, device)
    print(f"distilRB-rand: mae: {testMae:.3f}, r: {testR:.3f}")
    
    # Eval KQV
    testMae, testR = evalRobertaRegressor(rbKQVReg, testLoader, device)
    print(f"distilRB-KQV: mae: {testMae:.3f}, r: {testR:.3f}")
    
    # Eval nores
    testMae, testR = evalRobertaRegressor(rbNoresReg, testLoader, device)
    print(f"distilRB-nores: mae: {testMae:.3f}, r: {testR:.3f}")