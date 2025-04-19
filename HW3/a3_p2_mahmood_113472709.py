import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Load datasets
boolq_dataset = load_dataset('google/boolq')
sst_dataset = load_dataset('stanfordnlp/sst')

################  Helper Functions  ################

def print_boolq_metrics(preds, true_labels):
    accuracy = sklearn.metrics.accuracy_score(true_labels, preds)
    f1 = sklearn.metrics.f1_score(true_labels, preds, average='macro')
    print(f"overall acc: {accuracy:.3f}, f1: {f1:.3f}")
    return accuracy, f1

def plot_losses(losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def prep_roberta_input(item, tokenizer, max_length=512):
    passage = item['passage']
    question = item['question']
    answer = 1 if item['answer'] else 0
    
    # Format input
    text = f"{passage}\n{question}?\n"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    return inputs, answer

def init_roberta_loader(dataset, tokenizer, batch_size=16, shuffle=True):
    input_ids = []
    att_masks = []
    labels = []
    
    for i in range(len(dataset)):
        inputs, answer = prep_roberta_input(dataset[i], tokenizer)
        input_ids.append(inputs["input_ids"])
        att_masks.append(inputs["attention_mask"])
        labels.append(answer)
    
    # Convert to tensors
    input_ids = torch.stack(input_ids)
    att_masks = torch.stack(att_masks)
    labels = torch.tensor(labels)
    
    tensor_dataset = torch.utils.data.TensorDataset(input_ids, att_masks, labels)
    
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

def prep_roberta_regression_input(item, tokenizer, max_length=512):
    sentence = item['sentence']
    label = item['label']
    
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    return inputs, label

def init_roberta_regression_loader(dataset, tokenizer, batch_size=16, shuffle=True):
    input_ids = []
    att_masks = []
    labels = []
    
    for i in range(len(dataset)):
        inputs, label = prep_roberta_regression_input(dataset[i], tokenizer)
        input_ids.append(inputs["input_ids"])
        att_masks.append(inputs["attention_mask"])
        labels.append(label)
    
    # Convert to tensors
    input_ids = torch.stack(input_ids)
    att_masks = torch.stack(att_masks)
    labels = torch.tensor(labels, dtype=torch.float)
    
    tensor_dataset = torch.utils.data.TensorDataset(input_ids, att_masks, labels)
    
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


################  Part 2.1  ################

class RobertaRegressionHead(nn.Module):
    """Head for sentence-level regression tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForRegression(nn.Module):
    def __init__(self, roberta_model):
        super().__init__()
        self.roberta = roberta_model
        self.config = self.roberta.config
        self.regression_head = RobertaRegressionHead(self.config)
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.regression_head(sequence_output)
        return logits.squeeze(-1)

def convert_to_distilRB_rand(model):
    """Function to initialize weights randomly for distilRB-rand"""
    def custom_init_weights(module):
        if isinstance(module, nn.Linear):
            # Initialize linear layers with random Gaussian weights
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm weights to 1
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    model.apply(custom_init_weights)
    
    return model

def convert_to_distilRB_KQV(model):
    """Function to convert distilRoberta to distilRB-KQV by sharing KQV weights in the last 2 transformer layers"""
    # Get the last two transformer blocks
    for i in range(model.config.num_hidden_layers - 2, model.config.num_hidden_layers):
        layer = model.roberta.encoder.layer[i]
        
        # Get the query, key, value weights and biases
        q_weight = layer.attention.self.query.weight
        k_weight = layer.attention.self.key.weight
        v_weight = layer.attention.self.value.weight
        
        q_bias = layer.attention.self.query.bias
        k_bias = layer.attention.self.key.bias
        v_bias = layer.attention.self.value.bias
        
        # Calculate the mean of query and key weights
        shared_weight = (q_weight + k_weight) / 2
        shared_bias = (q_bias + k_bias) / 2
        
        # Create a new shared linear layer for KQV
        hidden_size = model.config.hidden_size
        shared_KQV = nn.Linear(hidden_size, hidden_size)
        shared_KQV.weight.data = shared_weight.clone()
        shared_KQV.bias.data = shared_bias.clone()
        
        # Replace the query, key, and value layers with the shared layer
        layer.attention.self.query = shared_KQV
        layer.attention.self.key = shared_KQV
        layer.attention.self.value = shared_KQV
    
    return model

def convert_to_distilRB_nores(model):
    """Function to convert distilRoberta to distilRB-nores by removing residual connections in the last 2 transformer layers"""
    
    # Monkey patch the forward method of the last two encoder layers
    for i in range(model.config.num_hidden_layers - 2, model.config.num_hidden_layers):
        layer = model.roberta.encoder.layer[i]
        
        # Store the original forward function
        orig_attention_forward = layer.attention.forward
        orig_output_forward = layer.output.forward
        
        # Define new forward functions without residual connections
        def new_attention_forward(self, hidden_states, attention_mask=None, head_mask=None, *args, **kwargs):
            # Original forward without the residual connection
            attention_outputs = orig_attention_forward(hidden_states, attention_mask, head_mask, *args, **kwargs)
            
            # No addition of residual connection here
            # return hidden_states + attention_outputs[0], attention_outputs[1:]
            return attention_outputs
            
        def new_output_forward(self, hidden_states, input_tensor):
            # Original forward without the residual connection
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            # No addition of residual connection
            hidden_states = self.LayerNorm(hidden_states)
            return hidden_states
            
        # Replace the forward methods
        layer.attention.forward = lambda *args, **kwargs: new_attention_forward(layer.attention, *args, **kwargs)
        layer.output.forward = lambda *args, **kwargs: new_output_forward(layer.output, *args, **kwargs)
        
        # Store the original layer forward
        orig_layer_forward = layer.forward
        
        # Define a new layer forward without residual connections
        def new_layer_forward(self, hidden_states, attention_mask=None, head_mask=None, *args, **kwargs):
            attention_outputs = self.attention(hidden_states, attention_mask, head_mask, *args, **kwargs)
            attention_output = attention_outputs[0]
            
            # No residual connection here
            outputs = attention_outputs[1:]
            
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            
            outputs = (layer_output,) + outputs
            return outputs
            
        # Replace the layer forward method
        layer.forward = lambda *args, **kwargs: new_layer_forward(layer, *args, **kwargs)
    
    return model

def get_model(variant='distilroberta', task_type='classifier'):
    """
    Function to instantiate distilRoberta model with/without modifications for classification and regression tasks
    Args:
    variant: distilroberta, distilRB-rand, distilRB-KQV, distilRB-nores
    task_type: classifier or regressor
    
    Returns:
    model: The distilRoberta model
    """
    if task_type == 'classifier':
        # Load the base model for classification
        model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
    else:
        # Load the base model for regression
        base_model = RobertaModel.from_pretrained("distilroberta-base")
        model = RobertaForRegression(base_model)
    
    # Apply the appropriate modifications based on the variant
    if variant == 'distilRB-rand':
        model = convert_to_distilRB_rand(model)
    elif variant == 'distilRB-KQV':
        model = convert_to_distilRB_KQV(model)
    elif variant == 'distilRB-nores':
        model = convert_to_distilRB_nores(model)
    # For 'distilroberta', no modification needed
    
    return model


################  Part 2.2  ################

def finetune_roberta_classifier(model, train_loader, num_epochs=1, lr=1e-5, weight_decay=1e-3, device='cuda'):
    model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Get total steps
    total_steps = len(train_loader) * num_epochs

    # LR scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    batch_losses = []
    
    for _ in range(num_epochs):
        for input_ids, att_mask, labels in tqdm.tqdm(train_loader):
            # Move batch to device
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
                loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())
            
    return model, batch_losses

def eval_roberta_classifier(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for input_ids, att_mask, labels in test_loader:
            # Move batch to device
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=att_mask)
            
            # Get predictions
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            
            preds.extend(pred)
            true_labels.extend(labels.numpy())
    
    return preds, true_labels


################  Part 2.3  ################

def finetune_roberta_regressor(model, train_loader, num_epochs=3, lr=2e-5, weight_decay=1e-3, device='cuda'):
    model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Get total steps
    total_steps = len(train_loader) * num_epochs

    # LR scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # MSE Loss
    criterion = nn.MSELoss()
    
    batch_losses = []
    
    for _ in range(num_epochs):
        for input_ids, att_mask, labels in tqdm.tqdm(train_loader):
            # Move batch to device
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=att_mask)
                loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())
            
    return model, batch_losses

def eval_roberta_regressor(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for input_ids, att_mask, labels in test_loader:
            # Move batch to device
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=att_mask)
            
            preds.extend(outputs.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    # Calculate metrics
    preds = np.array(preds)
    true_labels = np.array(true_labels)
    
    mae = np.mean(np.abs(preds - true_labels))   
    r = np.corrcoef(preds, true_labels)[0, 1]
    
    return mae, r


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
    
    # ----------------------- Checkpoint 2.2 -----------------------
    print("\nCheckpoint 2.2 - BoolQ Classification Task")
    
    # Initialize dataloaders for classification task
    train_loader = init_roberta_loader(boolq_dataset['train'], tokenizer)
    val_loader = init_roberta_loader(boolq_dataset['validation'], tokenizer, batch_size=16, shuffle=False)
    
    # Define models for each variant
    print("Initializing models...")
    distilRB_rand = get_model(variant='distilRB-rand', task_type='classifier')
    distilroberta = get_model(variant='distilroberta', task_type='classifier')
    distilRB_KQV = get_model(variant='distilRB-KQV', task_type='classifier')
    distilRB_nores = get_model(variant='distilRB-nores', task_type='classifier')
    
    # Fine-tune distilroberta model
    print("\nFine-tuning distilroberta model...")
    distilroberta, roberta_losses = finetune_roberta_classifier(distilroberta, train_loader)
    
    # Fine-tune distilRB-KQV model
    print("\nFine-tuning distilRB-KQV model...")
    distilRB_KQV, kqv_losses = finetune_roberta_classifier(distilRB_KQV, train_loader)
    
    # Fine-tune distilRB-nores model
    print("\nFine-tuning distilRB-nores model...")
    distilRB_nores, nores_losses = finetune_roberta_classifier(distilRB_nores, train_loader)
    
    # Plot loss curves
    plot_losses(roberta_losses, "DistilRoBERTa Loss Curve", "distilroberta_loss_curve.png")
    plot_losses(kqv_losses, "DistilRB-KQV Loss Curve", "distilRB_KQV_loss_curve.png")
    plot_losses(nores_losses, "DistilRB-nores Loss Curve", "distilRB_nores_loss_curve.png")
    
    # Evaluate models on validation set
    print("\nEvaluating models on BoolQ validation set...")
    
    # Evaluate distilRB-rand (no fine-tuning)
    print("distilRB-rand: ", end="")
    preds, true_labels = eval_roberta_classifier(distilRB_rand, val_loader, device)
    print_boolq_metrics(preds, true_labels)
    
    # Evaluate distilroberta
    print("distilroberta: ", end="")
    preds, true_labels = eval_roberta_classifier(distilroberta, val_loader, device)
    print_boolq_metrics(preds, true_labels)
    
    # Evaluate distilRB-KQV
    print("distilRB-KQV: ", end="")
    preds, true_labels = eval_roberta_classifier(distilRB_KQV, val_loader, device)
    print_boolq_metrics(preds, true_labels)
    
    # Evaluate distilRB-nores
    print("distilRB-nores: ", end="")
    preds, true_labels = eval_roberta_classifier(distilRB_nores, val_loader, device)
    print_boolq_metrics(preds, true_labels)
    
    # ----------------------- Checkpoint 2.3 -----------------------
    print("\nCheckpoint 2.3 - SST Regression Task")
    
    # Initialize dataloaders for regression task
    train_loader = init_roberta_regression_loader(sst_dataset['train'], tokenizer)
    val_loader = init_roberta_regression_loader(sst_dataset['validation'], tokenizer, batch_size=16, shuffle=False)
    test_loader = init_roberta_regression_loader(sst_dataset['test'], tokenizer, batch_size=16, shuffle=False)
    
    # Define models for each variant
    print("Initializing regression models...")
    reg_distilRB_rand = get_model(variant='distilRB-rand', task_type='regressor')
    reg_distilroberta = get_model(variant='distilroberta', task_type='regressor')
    reg_distilRB_KQV = get_model(variant='distilRB-KQV', task_type='regressor')
    reg_distilRB_nores = get_model(variant='distilRB-nores', task_type='regressor')
    
    # Fine-tune distilroberta model
    print("\nFine-tuning distilroberta model for regression...")
    reg_distilroberta, roberta_losses = finetune_roberta_regressor(reg_distilroberta, train_loader)
    
    # Plot loss curve
    plot_losses(roberta_losses, "DistilRoBERTa Regression Loss Curve", "distilroberta_regression_loss_curve.png")
    
    # Fine-tune distilRB-KQV model
    print("\nFine-tuning distilRB-KQV model for regression...")
    reg_distilRB_KQV, _ = finetune_roberta_regressor(reg_distilRB_KQV, train_loader)
    
    # Fine-tune distilRB-nores model
    print("\nFine-tuning distilRB-nores model for regression...")
    reg_distilRB_nores, _ = finetune_roberta_regressor(reg_distilRB_nores, train_loader)
    
    # Evaluate models
    print("\nEvaluating models on SST validation set...")
    
    # Evaluate distilroberta on validation and test sets
    val_mae, val_r = eval_roberta_regressor(reg_distilroberta, val_loader, device)
    test_mae, test_r = eval_roberta_regressor(reg_distilroberta, test_loader, device)
    print(f"For distilRoberta:")
    print(f"Validation: mae: {val_mae:.3f}, r: {val_r:.3f}")
    print(f"Test: mae: {test_mae:.3f}, r: {test_r:.3f}")
    
    # Evaluate other models on test set
    print("\nSST test set:")
    
    # Evaluate distilRB-rand
    test_mae, test_r = eval_roberta_regressor(reg_distilRB_rand, test_loader, device)
    print(f"distilRB-rand: mae: {test_mae:.3f}, r: {test_r:.3f}")
    
    # Evaluate distilRB-KQV
    test_mae, test_r = eval_roberta_regressor(reg_distilRB_KQV, test_loader, device)
    print(f"distilRB-KQV: mae: {test_mae:.3f}, r: {test_r:.3f}")
    
    # Evaluate distilRB-nores
    test_mae, test_r = eval_roberta_regressor(reg_distilRB_nores, test_loader, device)
    print(f"distilRB-nores: mae: {test_mae:.3f}, r: {test_r:.3f}")