import argparse
import os
import time
import copy
import types
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# PyTorch
import torch
import torchmetrics
# Hugging Face
import datasets
import transformers
# Importing our custom module(s)
import layers
import losses
import utils

def get_news_ag_datasets(dataset_directory, n, tune, random_state):

    ag_news_dataset = datasets.load_dataset('ag_news', cache_dir=dataset_directory)
    full_train_dataset = pd.DataFrame(ag_news_dataset['train'])
    full_test_dataset = pd.DataFrame(ag_news_dataset['test'])

    if n == len(full_train_dataset):
        train_and_val_indices = np.arange(0, len(full_train_dataset))
    else:
        train_and_val_indices, _ = train_test_split(
            np.arange(0, len(full_train_dataset)), 
            test_size=None, 
            train_size=n, 
            random_state=random_state, 
            shuffle=True, 
            stratify=np.array(full_train_dataset.label),
        )
        
    val_size = int((1/5) * n)
    train_indices, val_indices = train_test_split(
        train_and_val_indices, 
        test_size=val_size, 
        train_size=n-val_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=np.array(full_train_dataset.label)[train_and_val_indices],
    )
    
    if tune:
        return datasets.DatasetDict({
            'train': datasets.Dataset.from_pandas(full_train_dataset.iloc[train_indices]),
            'val_or_test': datasets.Dataset.from_pandas(full_train_dataset.iloc[val_indices])
        })
    else:
        return datasets.DatasetDict({
            'train': datasets.Dataset.from_pandas(full_train_dataset.iloc[train_and_val_indices]),
            'val_or_test': datasets.Dataset.from_pandas(full_test_dataset)
        })
    
def flatten_params(model, excluded_params=['lengthscale_param', 'noise_param', 'outputscale_param', 'sigma_param']):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])
    
def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None, num_classes=10, num_samples=1):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    #acc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc': 0.0, 'labels': [], 'lambda': 0.0, 'logits': [], 'loss': 0.0, 'nll': 0.0, 'tau': 0.0}
                    
    for batch in dataloader:
        
        batch_size = len(batch)
                                
        if device.type == 'cuda':
            batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()
        params = flatten_params(model)
        for sample_index in range(num_samples):
            labels = batch.pop('label')
            logits = model(**batch).logits
            losses = criterion(labels, logits, params, N=len(dataloader.dataset))
            losses['loss'].backward()
            
        # TODO: Average metrics over num_samples instead of returning metrics for last sample.
        if num_samples > 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(1/num_samples)
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if lr_scheduler:
            lr_scheduler.step()

        metrics['loss'] += (batch_size/dataset_size)*losses['loss'].item()
        metrics['nll'] += (batch_size/dataset_size)*losses['nll'].item()
        
        # Note: Added for DE ELBo
        metrics['lambda_star'] = losses.get('lambda_star')
        metrics['tau_star'] = losses.get('tau_star')
                    
        if device.type == 'cuda':
            labels, logits = labels.detach().cpu(), logits.detach().cpu()

        for label, logit in zip(labels, logits):
            metrics['labels'].append(label)
            metrics['logits'].append(logit)
                
    labels = torch.stack(metrics['labels'])
    logits = torch.stack(metrics['logits'])
    metrics['acc'] = acc(logits, labels).item()
            
    return metrics

def evaluate(model, criterion, dataloader, num_classes=10):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()
    
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    #acc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc': 0.0, 'labels': [], 'logits': [], 'loss': 0.0, 'nll': 0.0}
            
    with torch.no_grad():
        for batch in dataloader:
            
            batch_size = len(batch)
                                
            if device.type == 'cuda':
                batch = {k: v.to(device) for k, v in batch.items()}
            
            params = flatten_params(model)
            labels = batch.pop('label')
            logits = model(**batch).logits
            losses = criterion(labels, logits, params, N=len(dataloader.dataset))

            metrics['loss'] += (batch_size/dataset_size)*losses['loss'].item()
            metrics['nll'] += (batch_size/dataset_size)*losses['nll'].item()

            if device.type == 'cuda':
                labels, logits = labels.detach().cpu(), logits.detach().cpu()
    
            for label, logit in zip(labels, logits):
                metrics['labels'].append(label)
                metrics['logits'].append(logit)

        labels = torch.stack(metrics['labels'])
        logits = torch.stack(metrics['logits'])
        metrics['acc'] = acc(logits, labels).item()
        
    return metrics

# python fine-tuning_AG_News.py --alpha=0.0001 --batch_size=32 --beta=0.0001 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/AG_News' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/tuned_AG_News' --lr_0=0.01 --model_name='l2-sp_alpha=0.0001_beta=0.0001_lr_0=0.01_n=400_random_state=1001' --n=400 --random_state=1001 --tune
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--alpha', default=0.01, help='Backbone weight decay (default: 0.01)', type=float)
    parser.add_argument('--batch_size', default=32, help='Batch size (default: 32)', type=int)
    parser.add_argument('--beta', default=0.01, help='Classifier head weight decay (default: 0.01)', type=float)
    parser.add_argument('--criterion', default='l2-sp', help='Criterion (default: \'l2-sp\')', type=str)
    parser.add_argument('--dataset_directory', default='', help='Directory to dataset (default: \'\')', type=str)
    parser.add_argument('--ELBo', action='store_true', default=False, help='Whether or not to learn regularization strength with the ELBo (default: False)')
    parser.add_argument('--experiments_directory', default='', help='Directory to save experiments (default: \'\')', type=str)
    parser.add_argument('--kappa', default=1.0, help='TODO (default: 1.0)', type=float)
    parser.add_argument('--lambd', default=1.0, help='Covariance scaling factor (default: 1.0)', type=float)
    parser.add_argument('--lr_0', default=0.1, help='Initial learning rate (default: 0.1)', type=float)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--n', default=1000, help='Number of training samples (default: 1000)', type=int)
    parser.add_argument('--num_workers', default=0, help='Number of workers (default: 0)', type=int)
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument('--tune', action='store_true', default=False, help='Whether validation or test set is used (default: False)')
    args = parser.parse_args()    
    
    torch.manual_seed(args.random_state)
    os.makedirs(args.experiments_directory, exist_ok=True)
    
    dataset = get_news_ag_datasets(args.dataset_directory, args.n, args.tune, args.random_state)

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_datasets['val_or_test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    augmented_train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=args.batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=args.batch_size)
    val_or_test_loader = torch.utils.data.DataLoader(tokenized_datasets['val_or_test'], batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    num_classes = 4
    model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    bb_loc = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.bert.parameters()).detach()).to(device)
    model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
    utils.add_variational_layers(model, model.sigma_param)
    model.use_posterior = types.MethodType(utils.use_posterior, model)
    #criterion = losses.L2SPLoss(args.alpha, bb_loc, args.beta)
    criterion = losses.L2KappaELBoLoss(bb_loc, args.kappa, model.sigma_param)

    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_0, weight_decay=0.0)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_0, weight_decay=0.0)

    steps = 12_000
    num_batches = len(augmented_train_loader)
    epochs = int(steps/num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)

    if not args.ELBo:
        columns = ['epoch', 'train_acc', 'train_loss', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']
    else:
        columns = ['epoch', 'lambda_star', 'sigma', 'tau_star', 'train_acc', 'train_loss', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']

    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(epochs):
        train_epoch_start_time = time.time()
        augmented_train_metrics = train_one_epoch(model, criterion, optimizer, augmented_train_loader, lr_scheduler=lr_scheduler, num_classes=num_classes)
        train_epoch_end_time = time.time()
        #train_metrics = evaluate(model, criterion, train_loader, num_classes=num_classes)
        train_metrics = augmented_train_metrics
        
        if args.tune or epoch == epochs-1:
            if not args.ELBo:
                val_or_test_metrics = evaluate(model, criterion, val_or_test_loader, num_classes=num_classes)
            else:
                model.use_posterior(True)
                num_samples = 10
                sample_metrics = [evaluate(model, criterion, train_loader, num_classes=num_classes) for _ in range(num_samples)]
                train_metrics = {key: sum(metrics[key] for metrics in sample_metrics) / num_samples for key in ['acc', 'loss', 'nll']}
                model.use_posterior(False)
                val_or_test_metrics = evaluate(model, criterion, val_or_test_loader, num_classes=num_classes)
        else:
            val_or_test_metrics = {'acc': 0.0, 'loss': 0.0, 'nll': 0.0}
                
        if not args.ELBo:
            row = [epoch, train_metrics['acc'], train_metrics['loss'], train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
        else:
            row = [epoch, augmented_train_metrics['lambda_star'].item(), torch.nn.functional.softplus(model.sigma_param).item(), augmented_train_metrics['tau_star'].item(), train_metrics['acc'], train_metrics['loss'], train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
            
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_directory}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_directory}/{args.model_name}.pt')
