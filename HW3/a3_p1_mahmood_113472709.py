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

def plotLosses(losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def prepare_gpt2_input(item, tokenizer, include_answer=False, max_length=1024):
    passage = item['passage']
    question = item['question']
    answer = "yes" if item['answer'] else "no"
    
    # Format the input text
    text = f"{passage}\n{question}?\n"
    
    if include_answer:
        # For training, include the answer
        full_text = text + answer
        # No padding here - we'll do dynamic padding in the collate function
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
    else:
        # For inference, just include the passage and question
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    return inputs, text, answer

def create_gpt2_dataloader(dataset, tokenizer, batch_size=8, shuffle=True):
    processed_data = []
    
    for i in range(len(dataset)):
        inputs, _, _ = prepare_gpt2_input(dataset[i], tokenizer, include_answer=True, max_length=1024)
        processed_data.append(inputs)
    
    # Create a dynamic padding collate function
    def collate_fn(batch):
        # Find the maximum sequence length in this specific batch
        max_batch_length = max(item["input_ids"].size(0) for item in batch)
        
        collated_batch = {}
        for key in batch[0].keys():
            if key in ["input_ids", "attention_mask", "labels"]:
                # Pad each tensor to the max length of this batch (not the global max)
                padded_tensors = []
                for item in batch:
                    tensor = item[key]
                    current_len = tensor.size(0)
                    if current_len < max_batch_length:
                        # Pad with appropriate values (0 for attention_mask, -100 for labels, pad_token_id for input_ids)
                        if key == "attention_mask":
                            padding_value = 0
                        elif key == "labels":
                            padding_value = -100
                        else:  # input_ids
                            padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                        
                        padded = torch.cat([
                            tensor,
                            torch.full((max_batch_length - current_len,), padding_value, dtype=tensor.dtype)
                        ])
                        padded_tensors.append(padded)
                    else:
                        padded_tensors.append(tensor)
                
                collated_batch[key] = torch.stack(padded_tensors)
        
        return collated_batch
    
    return DataLoader(processed_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def prepare_roberta_input(item, tokenizer, max_length=512):
    passage = item['passage']
    question = item['question']
    answer = 1 if item['answer'] else 0  # 1 for yes, 0 for no
    
    # Format the input text
    text = f"{passage}\n{question}?\n"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    return inputs, answer

def create_roberta_dataloader(dataset, tokenizer, batch_size=16, shuffle=True):
    input_ids = []
    attention_masks = []
    labels = []
    
    for i in range(len(dataset)):
        inputs, answer = prepare_roberta_input(dataset[i], tokenizer)
        input_ids.append(inputs["input_ids"])
        attention_masks.append(inputs["attention_mask"])
        labels.append(answer)
    
    # Convert to tensors
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    
    # Create TensorDataset
    tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
    
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


# ------------Part 1.1------------------

def evalGPT2(model, dataset, tokenizer, device):
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
        yesID = tokenizer.encode("yes")[0]
        noID = tokenizer.encode("no")[0]

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

# ------------Part 1.2------------------

def finetuneGPT2(model, train_loader, num_epochs=1, lr=1e-5, weight_decay=1e-3, device='cuda'):
    model.to(device)
    model.train()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Calculate total training steps
    totalSteps = len(train_loader) * num_epochs
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=totalSteps)
    
    batch_losses = []
    
    for epoch in range(num_epochs):        
        for batch in tqdm.tqdm(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
              outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate schedule
            scheduler.step()
            
            batch_losses.append(loss.item())
            
    return model, batch_losses

# ------------Part 1.4------------------

def finetuneRoberta(model, train_loader, num_epochs=1, lr=1e-5, weight_decay=1e-3, device='cuda'):
    model.to(device)
    model.train()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Calculate total training steps
    totalSteps = len(train_loader) * num_epochs
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=totalSteps)
    
    batch_losses = []
    
    for epoch in range(num_epochs):
        for batch_input_ids, batch_attention_mask, batch_labels in tqdm.tqdm(train_loader):
            # Move batch to device
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
              outputs = model(input_ids=batch_input_ids,attention_mask=batch_attention_mask,labels=batch_labels)
            
            loss = outputs.loss            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate schedule
            scheduler.step()
            
            batch_losses.append(loss.item())
    return model, batch_losses

def evalRoberta(model, data_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in data_loader:
            # Move batch to device
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids=batch_input_ids,attention_mask=batch_attention_mask)

            # Get predictions
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            
            preds.extend(pred)
            true_labels.extend(batch_labels.numpy())
    
    # Convert predictions to boolean (1 = "yes", 0 = "no")
    preds_bool = [bool(p) for p in preds]
    true_labels_bool = [bool(l) for l in true_labels]
    
    return preds_bool, true_labels_bool

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Checkpoint 1.1
    print("Checkpoint 1.1")

    # Load model, tokenizer, dataset
    gpt2Tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    gpt2Model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    gpt2Model.resize_token_embeddings(len(gpt2Tokenizer))
    if gpt2Tokenizer.pad_token is None:
        gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token

    preds, trueLabels = evalGPT2(gpt2Model, boolq_dataset['validation'], gpt2Tokenizer, device)

    # Get metrics and print them
    printMetrics(preds, trueLabels)

    # Clear memory so it doesn't run out and give CUDA error on colab
    torch.cuda.empty_cache()

    # Checkpoint 1.2
    print("Checkpoint 1.2")

    # Create train dataloader for GPT2
    train_loader = create_gpt2_dataloader(boolq_dataset['train'], gpt2Tokenizer, batch_size=8)
    
    # Fine tune the model
    gpt2ModelFinetuned, batchLosses = finetuneGPT2(gpt2Model, train_loader)

    # Plot loss curve
    plotLosses(batchLosses, "GPT2 Loss Curve", "gpt2_loss_curve.png")
  
    # Checkpoint 1.3
    print("Checkpoint 1.3")
    torch.cuda.empty_cache()
    
    # Evaluate the fine-tuned model
    preds, trueLabels = evalGPT2(gpt2ModelFinetuned, boolq_dataset['validation'], gpt2Tokenizer, device)
    # Print metrics
    printMetrics(preds, trueLabels)

    # Checkpoint 1.4
    print("Checkpoint 1.4")
    torch.cuda.empty_cache()

    # Load RoBERTa model and tokenizer
    roberta_tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
    roberta_model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
    
    # Create dataloaders for RoBERTa
    roberta_train_loader = create_roberta_dataloader(boolq_dataset['train'], roberta_tokenizer, batch_size=16)
    roberta_val_loader = create_roberta_dataloader(boolq_dataset['validation'], roberta_tokenizer, batch_size=16, shuffle=False)
    
    # Fine-tune RoBERTa
    roberta_model_finetuned, roberta_losses = finetuneRoberta(roberta_model,roberta_train_loader)
    
    # Plot loss curve
    # use helper function to plot losses
    plotLosses(roberta_losses, "RoBERTa Loss Curve", "roberta_loss_curve.png")

    # Evaluate RoBERTa
    preds, trueLabels = evalRoberta(roberta_model_finetuned, roberta_val_loader, device)
    
    # Print metrics
    printMetrics(preds,trueLabels)
    torch.cuda.empty_cache()

