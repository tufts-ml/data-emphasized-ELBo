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
import likelihoods
import models
import losses
import priors
import utils

def get_news_ag_datasets(dataset_directory, n, tune, random_state):

    ag_news_dataset = datasets.load_dataset("ag_news", cache_dir=dataset_directory)
    full_train_dataset = pd.DataFrame(ag_news_dataset["train"])
    full_test_dataset = pd.DataFrame(ag_news_dataset["test"])

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
            "train": datasets.Dataset.from_pandas(full_train_dataset.iloc[train_indices]),
            "val_or_test": datasets.Dataset.from_pandas(full_train_dataset.iloc[val_indices])
        })
    else:
        return datasets.DatasetDict({
            "train": datasets.Dataset.from_pandas(full_train_dataset.iloc[train_and_val_indices]),
            "val_or_test": datasets.Dataset.from_pandas(full_test_dataset)
        })
    
def flatten_params(model, excluded_params=["raw_lengthscale", "raw_noise", "raw_outputscale", "raw_sigma", "raw_tau"]):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None, num_samples=1):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.train()

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {"probs": [], "labels": []}
    
    for batch in dataloader:
        
        batch_size = len(batch)
                                
        if device.type == 'cuda':
            batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        params = flatten_params(model)

        sample_probs = []
        for _ in range(num_samples):

            labels = batch.pop("label")
            logits = model(**batch).logits
            losses = criterion(logits, labels, params, len(dataloader.dataset))
            losses["loss"].backward()

            for key, value in losses.items():
                metrics[key] = metrics.get(key, 0.0) + (batch_size / dataset_size) * (1 / num_samples) * value.item()
                
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sample_probs.append(probs.detach().cpu().numpy())

        if num_samples > 1:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.mul_(1/num_samples)

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            
        optimizer.step()
        
        if lr_scheduler:
            lr_scheduler.step()
            
        metrics["probs"].append(np.stack(sample_probs, axis=1))
        metrics["labels"].append(labels.detach().cpu().numpy())
        
    metrics["probs"] = np.concatenate(metrics["probs"], axis=0)
    metrics["labels"] = np.concatenate(metrics["labels"], axis=0)

    dataset_size, num_samples, num_classes = metrics["probs"].shape
    bma_preds = metrics["probs"].mean(axis=1).argmax(axis=-1)
    metrics["acc"] = np.mean([(bma_preds[metrics["labels"] == c] == c).mean() for c in range(num_classes)])
                    
    return metrics

def evaluate(model, criterion, dataloader, num_samples=1):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.eval()

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {"probs": [], "labels": []}
    
    with torch.no_grad():
        for batch in dataloader:

            batch_size = len(batch)

            if device.type == 'cuda':
                batch = {k: v.to(device) for k, v in batch.items()}

            params = flatten_params(model)

            sample_probs = []
            for _ in range(num_samples):

                labels = batch.pop("label")
                logits = model(**batch).logits
                losses = criterion(logits, labels, params, len(dataloader.dataset))

                for key, value in losses.items():
                    metrics[key] = metrics.get(key, 0.0) + (batch_size / dataset_size) * (1 / num_samples) * value.item()
                    
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sample_probs.append(probs.detach().cpu().numpy())
                
            metrics["probs"].append(np.stack(sample_probs, axis=1))
            metrics["labels"].append(labels.detach().cpu().numpy())
            
        metrics["probs"] = np.concatenate(metrics["probs"], axis=0)
        metrics["labels"] = np.concatenate(metrics["labels"], axis=0)
        
        dataset_size, num_samples, num_classes = metrics["probs"].shape
        bma_preds = metrics["probs"].mean(axis=1).argmax(axis=-1)
        metrics["acc"] = np.mean([(bma_preds[metrics["labels"] == c] == c).mean() for c in range(num_classes)])
        
    return metrics

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--alpha", default=1.0, help="Alpha (default: 1.0)", type=float)
    parser.add_argument("--batch_size", default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument("--beta", default=1.0, help="Beta (default: 1.0)", type=float)
    parser.add_argument("--dataset", default="", help="Dataset (default: \"\")", type=str)
    parser.add_argument("--dataset_dir", default="", help="Directory to dataset (default: \"\")", type=str)
    parser.add_argument("--experiments_dir", default="", help="Directory to save experiments (default: \"\")", type=str)
    parser.add_argument("--k", default=5, help="Rank of PTYL prior (default: 5)", type=int)
    parser.add_argument("--kappa", default=1.0, help="Kappa (default: 1.0)", type=float)
    parser.add_argument("--lambd", default=1.0, help="Lambda (default: 1.0)", type=float)
    parser.add_argument("--lr_0", default=0.1, help="Initial learning rate (default: 0.1)", type=float)
    parser.add_argument("--method", default="Laplace", help="Method (default: \"Laplace\")", type=str)
    parser.add_argument("--model", default="l2-sp", help="Model (default: \"l2-sp\")", type=str)
    parser.add_argument("--model_arch", default="BERT-base", help="Model architecture (default: \"BERT-base\")", type=str)
    parser.add_argument("--model_name", default="test", help="Model name (default: \"test\")", type=str)
    parser.add_argument("--n", default=1000, help="Number of training samples (default: 1000)", type=int)
    parser.add_argument("--num_workers", default=0, help="Number of workers (default: 0)", type=int)
    parser.add_argument("--prior_dir", default="", help="Directory to prior (default: \"\")", type=str)
    parser.add_argument("--prior_type", default="", help="Prior type (default: \"\")", type=str)
    parser.add_argument("--random_state", default=42, help="Random state (default: 42)", type=int)
    parser.add_argument("--save", action="store_true", default=False, help="Whether or not to save the model (default: False)")
    parser.add_argument("--tune", action="store_true", default=False, help="Whether validation or test set is used (default: False)")
    args = parser.parse_args()
    
    torch.manual_seed(args.random_state)
    
    os.makedirs(args.experiments_dir, exist_ok=True)
    
    assert args.dataset in ["News-4"]
    if args.dataset == "News-4":
        num_classes = 4
        dataset = get_news_ag_datasets(args.dataset_dir, args.n, args.tune, args.random_state)

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_datasets["val_or_test"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    augmented_train_dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
    la_train_dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], batch_size=32)
    train_dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], batch_size=32)
    val_or_test_dataloader = torch.utils.data.DataLoader(tokenized_datasets["val_or_test"], batch_size=32)
    
    assert args.model_arch in ["BERT-base"]
    if args.model_arch == "BERT-base":
        model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        num_classifier_params = (768 * num_classes) + num_classes
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model.to(device)
        
    assert args.model in ["l2-zero", "l2-sp"]
    if args.model == "l2-zero":
        backbone_prior_params = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.bert.parameters()).detach()).to(device)
        backbone_prior_params = torch.zeros_like(backbone_prior_params)
        backbone_prior = priors.IsotropicGaussianPrior(num_params=len(backbone_prior_params), prior_variance=1.0)
        classifier_prior = priors.IsotropicGaussianPrior(num_params=num_classifier_params, prior_variance=1.0)
    elif args.model == "l2-sp":
        backbone_prior_params = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.bert.parameters()).detach()).to(device)
        backbone_prior = priors.IsotropicGaussianPrior(prior_params=backbone_prior_params, prior_variance=1.0)
        classifier_prior = priors.IsotropicGaussianPrior(num_params=num_classifier_params, prior_variance=1.0)
        
    likelihood = likelihoods.CategoricalLikelihood(num_classes=num_classes)
    
    backbone_prior.to(device)
    classifier_prior.to(device)
    likelihood.to(device)
        
    assert args.method in ["Laplace"]
    if args.method == "Laplace":
        prior_precision = torch.tensor([1.0, 1.0], device=device)
        classifier_prior_params = torch.zeros(size=(num_classifier_params,), device=device)
        prior_mean = torch.cat((backbone_prior_params, classifier_prior_params))
        la = models.L2SPLaplace(model=model, likelihood="classification", prior_precision=prior_precision, prior_mean=prior_mean, backend=laplace.curvature.AsdlEF)
        la.num_backbone_params = len(backbone_prior_params)    
        criterion = losses.TransferLearningMAPLoss(model, likelihood, backbone_prior, classifier_prior)

    optimizer = torch.optim.SGD([{"params": model.parameters()}, {"params": likelihood.parameters()}, {"params": backbone_prior.parameters()}, {"params": classifier_prior.parameters()}], lr=args.lr_0, weight_decay=0.0, momentum=0.9, nesterov=True)

    steps = 12000
    num_batches = len(augmented_train_dataloader)
    epochs = int(steps / num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)
    
    B, F, K, gamma = 0, 1, 100, 1.0
    
    for epoch in range(epochs):
        train_epoch_start_time = time.time()
        
        augmented_train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_dataloader, lr_scheduler=lr_scheduler, num_samples=1)
                
        if (epoch + 1) > B and (epoch + 1) % F == 0:
            la.fit(la_train_dataloader)
            la.optimize_prior_precision(pred_type="glm", method="marglik", n_steps=K, lr=gamma, init_prior_prec=la.prior_precision)
            backbone_prior.prior_variance = 1 / la.prior_precision[0]
            classifier_prior.prior_variance = 1 / la.prior_precision[1]
                        
        train_epoch_end_time = time.time()
        #train_metrics = evaluate(model, criterion, train_dataloader, num_classes=num_classes)
        train_metrics = augmented_train_metrics
        
        if args.tune or epoch == epochs - 1:
            val_or_test_metrics = evaluate(model, criterion, val_or_test_dataloader, num_samples=1)
        else:
            val_or_test_metrics = {key: 0.0 for key, value in train_metrics.items()}
            
        train_metrics["lml"] = la.log_marginal_likelihood().item()
          
        if epoch == 0:
            columns = ["epoch", *[f"train_{key}" for key in train_metrics.keys() if key not in ["probs", "labels"]], "train_sec/epoch", *[f"val_or_test_{key}" for key in val_or_test_metrics.keys() if key not in ["probs", "labels"]]]
            model_history_df = pd.DataFrame(columns=columns)
            
        row = [epoch, *[value for key, value in train_metrics.items() if key not in ["probs", "labels"]], train_epoch_end_time - train_epoch_start_time, *[value for key, value in val_or_test_metrics.items() if key not in ["probs", "labels"]]]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_dir}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_dir}/{args.model_name}.pt')
