import argparse
import os
import time
import copy
import types
import typing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# PyTorch
import torch
import torchmetrics
# Hugging Face
import datasets
import transformers
# Laplace
from laplace.baselaplace import DiagLaplace
from laplace.curvature import AsdlEF
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
    
class L2SPLaplace(DiagLaplace):
    
    @property
    def prior_precision(self) -> torch.Tensor:
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision: float | torch.Tensor):
        self._posterior_scale = None
        self._prior_precision = prior_precision.to(
            device=self._device, dtype=self._dtype
        )
        
    @property
    def prior_precision_diag(self) -> torch.Tensor:
        alpha = self.prior_precision[0]
        beta = self.prior_precision[1]
        return torch.cat([alpha * torch.ones(self.D, device=self._device, dtype=self._dtype),
                          beta * torch.ones(self.n_params - self.D, device=self._device, dtype=self._dtype)])
    
class MyBERT(torch.nn.Module):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, num_labels: int):
        super().__init__()
        config = transformers.BertConfig.from_pretrained("bert-base-uncased")
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = num_labels
        self.hf_model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)

    def forward(self, data: typing.MutableMapping) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attn_mask = data["attention_mask"].to(device)
        output_dict = self.hf_model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.logits
    
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
            labels = batch.pop('labels')
            logits = model(batch)
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
            labels = batch.pop('labels')
            logits = model(batch)
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

# python ../src/fine-tuning_AG_News-Copy1.py --batch_size=32 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/AG_News' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/tuned_AG_News_diagEF_SGD' --lr_0=0.1 --model_name='l2-sp_lr_0=0.1_n=4000_random_state=1001' --n=4000 --num_workers=0 --random_state=1001 --save
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
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    tokenized_datasets['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_datasets['val_or_test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    augmented_train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=args.batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=args.batch_size)
    val_or_test_loader = torch.utils.data.DataLoader(tokenized_datasets['val_or_test'], batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    num_classes = 4
    #model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    #model.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4, device=device))))
    #utils.add_variational_layers(model, model.sigma_param)
    #model.use_posterior = types.MethodType(utils.use_posterior, model)
    model = MyBERT(tokenizer, num_classes)
    model.to(device)
    
    prior_precision = torch.tensor([1.0, 1.0]).to(device)
    bb_loc = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.hf_model.bert.parameters()).detach()).to(device)
    clf_loc = torch.zeros((768 * num_classes) + num_classes).to(device)
    prior_mean = torch.cat([bb_loc, clf_loc])
    la = L2SPLaplace(model=model, likelihood='classification', prior_precision=prior_precision, prior_mean=prior_mean, backend=AsdlEF)
    la.D = len(bb_loc)
    alpha = la.prior_precision[0].item() / len(train_loader.dataset)
    beta = la.prior_precision[1].item() / len(train_loader.dataset)
    criterion = losses.L2SPLoss(alpha, bb_loc, beta)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_0, weight_decay=0.0)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_0, weight_decay=0.0)

    steps = 12000
    num_batches = len(augmented_train_loader)
    epochs = int(steps/num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)
    
    F, K, gamma = 1, 100, 1

    columns = ['epoch', 'alpha', 'beta', 'train_acc', 'train_loss', 'train_log_marglik', 'train_nll', 'train_sec/epoch', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll']
    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(epochs):
        train_epoch_start_time = time.time()
        
        augmented_train_metrics = train_one_epoch(model, criterion, optimizer, augmented_train_loader, lr_scheduler=lr_scheduler, num_classes=num_classes)
        
        if epoch % F == 0:
            la.fit(train_loader)
            la.optimize_prior_precision(pred_type='glm', method='marglik', n_steps=K, lr=gamma, init_prior_prec=la.prior_precision)
            criterion.alpha = la.prior_precision[0].item() / len(train_loader.dataset)
            criterion.beta = la.prior_precision[1].item() / len(train_loader.dataset)
        
        train_epoch_end_time = time.time()
        #train_metrics = evaluate(model, criterion, train_loader, num_classes=num_classes)
        train_metrics = augmented_train_metrics
        
        if args.tune or epoch == epochs-1:
            val_or_test_metrics = evaluate(model, criterion, val_or_test_loader, num_classes=num_classes)
        else:
            val_or_test_metrics = {'acc': 0.0, 'loss': 0.0, 'nll': 0.0}
                
        row = [epoch, criterion.alpha, criterion.beta, train_metrics['acc'], train_metrics['loss'], la.log_marginal_likelihood().item(), train_metrics['nll'], train_epoch_end_time-train_epoch_start_time, val_or_test_metrics['acc'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_directory}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_directory}/{args.model_name}.pt')
