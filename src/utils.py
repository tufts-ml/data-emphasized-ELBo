import os
import numpy as np
from sklearn.model_selection import train_test_split
# PyTorch
import torch
import torchvision
import torchmetrics
# Importing our custom module(s)
import layers

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def worker_init_fn(worker_id):
    # This worker initialization function sets CPU affinity for each worker to 
    # all available CPUs, significantly improving GPU utilization when using 
    # num_workers > 0 (see https://github.com/pytorch/pytorch/issues/99625).
    os.sched_setaffinity(0, range(os.cpu_count()))
    
def get_mean_and_std(dataset, indices, dims=(1, 2)):
    
    means, stds = [], []

    for image, label in map(dataset.__getitem__, indices):
        means.append(torch.mean(image, dim=dims).tolist())
        stds.append(torch.std(image, dim=dims).tolist())

    return torch.tensor(means).mean(dim=0), torch.tensor(stds).mean(dim=0)

class TensorSubset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, indices, transform=None):
        X, y = zip(*[dataset[i] for i in indices])
        self.X = torch.stack(X)
        self.y = torch.tensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.transform(self.X[index]), self.y[index]) if self.transform else (self.X[index], self.y[index])
    
def get_cifar10_datasets(dataset_directory, n, tune, random_state):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    full_train_dataset = torchvision.datasets.CIFAR10(root=dataset_directory, train=True, transform=transform, download=True)
    full_test_dataset = torchvision.datasets.CIFAR10(root=dataset_directory, train=False, transform=transform, download=True)

    if n == len(full_train_dataset):
        train_and_val_indices = np.arange(0, len(full_train_dataset))
    else:
        train_and_val_indices, _ = train_test_split(
            np.arange(0, len(full_train_dataset)), 
            test_size=None, 
            train_size=n, 
            random_state=random_state, 
            shuffle=True, 
            stratify=np.array(full_train_dataset.targets),
        )
        
    val_size = int((1/5) * n)
    train_indices, val_indices = train_test_split(
        train_and_val_indices, 
        test_size=val_size, 
        train_size=n-val_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=np.array(full_train_dataset.targets)[train_and_val_indices],
    )

    if tune:
        mean, std = get_mean_and_std(full_train_dataset, train_indices)
    else:
        mean, std = get_mean_and_std(full_train_dataset, train_and_val_indices)

    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])

    if tune:
        augmented_train_dataset = TensorSubset(full_train_dataset, train_indices, augmented_transform)
        train_dataset = TensorSubset(full_train_dataset, train_indices, transform)
        val_dataset = TensorSubset(full_train_dataset, val_indices, transform)
        return augmented_train_dataset, train_dataset, val_dataset
    else:
        augmented_train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, augmented_transform)
        train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, transform)
        test_dataset = TensorSubset(full_test_dataset, range(len(full_test_dataset)), transform)
        return augmented_train_and_val_dataset, train_and_val_dataset, test_dataset
    
def get_oxfordiiit_pet_datasets(dataset_directory, n, tune, random_state):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(256, 256)),
    ])
    full_train_dataset = torchvision.datasets.OxfordIIITPet(root=dataset_directory, split='trainval', transform=transform, download=True)
    full_test_dataset = torchvision.datasets.OxfordIIITPet(root=dataset_directory, split='test', transform=transform, download=True)
    
    np_random_state = np.random.RandomState(random_state)
    selected_indices = np.array([index for label in set(full_train_dataset._labels) for index in np_random_state.choice([index_i for index_i, label_i in zip(np.arange(0, len(full_train_dataset)), np.array(full_train_dataset._labels)) if label_i == label], size=93, replace=False)])

    if n == len(selected_indices):
        train_and_val_indices = np.arange(0, len(selected_indices))
    else:
        train_and_val_indices, _ = train_test_split(
            selected_indices, 
            test_size=None, 
            train_size=n, 
            random_state=random_state, 
            shuffle=True, 
            stratify=np.array(full_train_dataset._labels)[selected_indices],
        )

    val_size = int((1/5) * n)
    train_indices, val_indices = train_test_split(
        train_and_val_indices, 
        test_size=val_size, 
        train_size=n-val_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=np.array(full_train_dataset._labels)[train_and_val_indices],
    )

    if tune:
        mean, std = get_mean_and_std(full_train_dataset, train_indices)
    else:
        mean, std = get_mean_and_std(full_train_dataset, train_and_val_indices)

    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])

    if tune:
        augmented_train_dataset = TensorSubset(full_train_dataset, train_indices, augmented_transform)
        train_dataset = TensorSubset(full_train_dataset, train_indices, transform)
        val_dataset = TensorSubset(full_train_dataset, val_indices, transform)
        return augmented_train_dataset, train_dataset, val_dataset
    else:
        augmented_train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, augmented_transform)
        train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, transform)
        test_dataset = TensorSubset(full_test_dataset, range(len(full_test_dataset)), transform)
        return augmented_train_and_val_dataset, train_and_val_dataset, test_dataset
    
def get_flowers_102_datasets(dataset_directory, n, tune, random_state):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(256, 256)),
    ])
    full_train_dataset = torchvision.datasets.Flowers102(root=dataset_directory, split='train', transform=transform, download=True)
    full_test_dataset = torchvision.datasets.Flowers102(root=dataset_directory, split='test', transform=transform, download=True)

    if n == len(full_train_dataset):
        train_and_val_indices = np.arange(0, len(full_train_dataset))
    else:
        train_and_val_indices, _ = train_test_split(
            np.arange(0, len(full_train_dataset)), 
            test_size=None, 
            train_size=n, 
            random_state=random_state, 
            shuffle=True, 
            stratify=np.array(full_train_dataset._labels),
        )

    val_size = int((1/5) * n)
    train_indices, val_indices = train_test_split(
        train_and_val_indices, 
        test_size=val_size, 
        train_size=n-val_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=np.array(full_train_dataset._labels)[train_and_val_indices],
    )

    if tune:
        mean, std = get_mean_and_std(full_train_dataset, train_indices)
    else:
        mean, std = get_mean_and_std(full_train_dataset, train_and_val_indices)

    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])

    if tune:
        augmented_train_dataset = TensorSubset(full_train_dataset, train_indices, augmented_transform)
        train_dataset = TensorSubset(full_train_dataset, train_indices, transform)
        val_dataset = TensorSubset(full_train_dataset, val_indices, transform)
        return augmented_train_dataset, train_dataset, val_dataset
    else:
        augmented_train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, augmented_transform)
        train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, transform)
        test_dataset = TensorSubset(full_test_dataset, range(len(full_test_dataset)), transform)
        return augmented_train_and_val_dataset, train_and_val_dataset, test_dataset

def get_fgvcaircraft_datasets(dataset_directory, n, tune, random_state):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(256, 256)),
    ])
    full_train_dataset = torchvision.datasets.FGVCAircraft(root=dataset_directory, split='trainval', transform=transform, download=True)
    full_test_dataset = torchvision.datasets.FGVCAircraft(root=dataset_directory, split='test', transform=transform, download=True)

    if n == len(full_train_dataset):
        train_and_val_indices = np.arange(0, len(full_train_dataset))
    else:
        train_and_val_indices, _ = train_test_split(
            np.arange(0, len(full_train_dataset)), 
            test_size=None, 
            train_size=n, 
            random_state=random_state, 
            shuffle=True, 
            stratify=np.array(full_train_dataset._labels),
        )

    val_size = int((1/5) * n)
    train_indices, val_indices = train_test_split(
        train_and_val_indices, 
        test_size=val_size, 
        train_size=n-val_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=np.array(full_train_dataset._labels)[train_and_val_indices],
    )

    if tune:
        mean, std = get_mean_and_std(full_train_dataset, train_indices)
    else:
        mean, std = get_mean_and_std(full_train_dataset, train_and_val_indices)

    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])

    if tune:
        augmented_train_dataset = TensorSubset(full_train_dataset, train_indices, augmented_transform)
        train_dataset = TensorSubset(full_train_dataset, train_indices, transform)
        val_dataset = TensorSubset(full_train_dataset, val_indices, transform)
        return augmented_train_dataset, train_dataset, val_dataset
    else:
        augmented_train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, augmented_transform)
        train_and_val_dataset = TensorSubset(full_train_dataset, train_and_val_indices, transform)
        test_dataset = TensorSubset(full_test_dataset, range(len(full_test_dataset)), transform)
        return augmented_train_and_val_dataset, train_and_val_dataset, test_dataset
    
def add_variational_layers(module, sigma_param):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, layers.VariationalLinear(child, sigma_param))
        elif isinstance(child, torch.nn.Conv2d):
            setattr(module, name, layers.VariationalConv2d(child, sigma_param))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, layers.VariationalBatchNorm2d(child, sigma_param))
        elif isinstance(child, torchvision.models.convnext.LayerNorm2d):
            setattr(module, name, layers.VariationalLayerNorm2d(child, sigma_param))
        elif isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, layers.VariationalLayerNorm(child, sigma_param))
        elif isinstance(child, torchvision.models.convnext.CNBlock):
            setattr(module, name, layers.VariationalCNBlock(child, sigma_param))
            add_variational_layers(child, sigma_param)
        elif isinstance(child, torch.nn.MultiheadAttention):
            setattr(module, name, layers.VariationalMultiheadAttention(child, sigma_param))
        elif isinstance(child, torch.nn.Embedding):
            setattr(module, name, layers.VariationalEmbedding(child, sigma_param))
        else:
            add_variational_layers(child, sigma_param)
            
def use_posterior(self, flag):
    for child in self.modules():
        if isinstance(child, (
            layers.VariationalLinear, 
            layers.VariationalConv2d, 
            layers.VariationalBatchNorm2d,
            layers.VariationalLayerNorm2d,
            layers.VariationalLayerNorm,
            layers.VariationalCNBlock,
        )):
            child.use_posterior = flag
    
def flatten_params(model, excluded_params=['lengthscale_param', 'noise_param', 'outputscale_param', 'sigma_param']):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])
    
def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None, num_classes=10, num_samples=1):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    #acc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc': 0.0, 'labels': [], 'lambda': 0.0, 'logits': [], 'loss': 0.0, 'nll': 0.0, 'tau': 0.0}
                    
    for images, labels in dataloader:
        
        batch_size = len(images)
                                
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        params = flatten_params(model)
        for sample_index in range(num_samples):
            logits = model(images)
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
        for images, labels in dataloader:
            
            batch_size = len(images)
                        
            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)
            
            params = flatten_params(model)
            logits = model(images)
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
