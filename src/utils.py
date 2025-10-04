import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
# PyTorch
import torch
import torchvision
import torchmetrics
# Importing our custom module(s)
import layers

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))
        
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
    
def get_flower102_datasets(dataset_directory, n, tune, random_state):

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
    
def get_pet37_datasets(dataset_directory, n, tune, random_state):

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
    
def add_variational_layers(module, raw_sigma):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, layers.VariationalLinear(child, raw_sigma))
        elif isinstance(child, torch.nn.Conv2d):
            setattr(module, name, layers.VariationalConv2d(child, raw_sigma))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, layers.VariationalBatchNorm2d(child, raw_sigma))
        elif isinstance(child, torchvision.models.convnext.LayerNorm2d):
            setattr(module, name, layers.VariationalLayerNorm2d(child, raw_sigma))
        elif isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, layers.VariationalLayerNorm(child, raw_sigma))
        elif isinstance(child, torchvision.models.convnext.CNBlock):
            setattr(module, name, layers.VariationalCNBlock(child, raw_sigma))
            add_variational_layers(child, raw_sigma)
        elif isinstance(child, torch.nn.MultiheadAttention):
            setattr(module, name, layers.VariationalMultiheadAttention(child, raw_sigma))
        elif isinstance(child, torch.nn.Embedding):
            setattr(module, name, layers.VariationalEmbedding(child, raw_sigma))
        else:
            add_variational_layers(child, raw_sigma)
            
def use_posterior(self, flag):
    for child in self.modules():
        if isinstance(child, (
            layers.VariationalLinear, 
            layers.VariationalConv2d, 
            layers.VariationalBatchNorm2d,
            layers.VariationalLayerNorm2d,
            layers.VariationalLayerNorm,
            layers.VariationalCNBlock,
            layers.VariationalMultiheadAttention,
            layers.VariationalEmbedding,
        )):
            child.use_posterior = flag
            
def assign_diag_raw_sigma(model, raw_sigma):
    idx = 0
    for child in model.modules():
        if hasattr(child, "_flatten"):
            num_params = len(child._flatten())
            raw_sigma_slice = raw_sigma[idx:idx+num_params]
            child.raw_sigma = raw_sigma_slice
            idx += num_params
            
def replace_layernorm2d(module):
    for name, child in module.named_children():
        if isinstance(child, torchvision.models.convnext.LayerNorm2d):
            device = next(child.parameters()).device
            new_norm = torch.nn.LayerNorm(
                child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine
            ).to(device)
            if child.elementwise_affine:
                new_norm.weight.data.copy_(child.weight.data)
                new_norm.bias.data.copy_(child.bias.data)

            new_child = torch.nn.Sequential(
                torchvision.ops.misc.Permute([0, 2, 3, 1]),
                new_norm,
                torchvision.ops.misc.Permute([0, 3, 1, 2]),
            )
            setattr(module, name, new_child)
        else:
            replace_layernorm2d(child)
            
def replace_cnblock(model):
    for stage_idx, stage in enumerate(model.features):
        for block_idx, block in enumerate(stage):
            if isinstance(block, torchvision.models.convnext.CNBlock):
                dim = block.layer_scale.shape[0]
                device = next(block.parameters()).device
                new_block = layers.CNBlock(
                    dim=dim,
                    layer_scale=1.0,
                    stochastic_depth_prob=block.stochastic_depth.p,
                    norm_layer=lambda _: torch.nn.LayerNorm(
                        normalized_shape=block.block[2].normalized_shape, 
                        eps=block.block[2].eps,
                        elementwise_affine=block.block[2].elementwise_affine
                    )
                ).to(device)
                new_block.block.load_state_dict(block.block.state_dict())
                new_block.layer_scale.weight.data = block.layer_scale.view(-1, 1, 1, 1)
                model.features[stage_idx][block_idx] = new_block
            
def flatten_params(model, excluded_params=["raw_lengthscale", "raw_noise", "raw_outputscale", "raw_sigma", "raw_tau"]):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])

def unflatten_params(model, params, excluded_params=["raw_lengthscale", "raw_noise", "raw_outputscale", "raw_sigma", "raw_tau"]):
    index = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name not in excluded_params:
            numel = param.numel()
            param.data.copy_(params[index:index+numel].view_as(param))
            index += numel

def flatten_grads(model, excluded_params=["raw_lengthscale", "raw_noise", "raw_outputscale", "raw_sigma", "raw_tau"]):
    return torch.cat([param.grad.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None, num_samples=1):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.train()

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {"probs": [], "labels": []}
    
    for images, labels in dataloader:
        
        batch_size = len(images)
                                
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        params = flatten_params(model)

        sample_probs = []
        for _ in range(num_samples):

            logits = model(images)
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
        for images, labels in dataloader:

            batch_size = len(images)

            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)

            params = flatten_params(model)

            sample_probs = []
            for _ in range(num_samples):

                logits = model(images)
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

def evaluate_clml(model, la, dataloader, num_samples=1, temp=1.0):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.eval()

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {"probs": [], "labels": []}
        
    with torch.no_grad():
    
        eps = torch.randn(size=(num_samples, la.n_params), device=device)
        samples = la.mean.reshape(1, la.n_params) + temp * la.posterior_scale.reshape(1, la.n_params) * eps
    
        for images, labels in dataloader:

            batch_size = len(images)

            if device.type == "cuda":
                images, labels = images.to(device), labels.to(device)

            sample_probs = []
            for sample in samples:

                torch.nn.utils.vector_to_parameters(sample, model.parameters())
                logits = model(images)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sample_probs.append(probs.detach().cpu().numpy())
                
            metrics["probs"].append(np.stack(sample_probs, axis=1))
            metrics["labels"].append(labels.detach().cpu().numpy())
            
        torch.nn.utils.vector_to_parameters(la.mean, model.parameters())
            
        metrics["probs"] = np.concatenate(metrics["probs"], axis=0)
        metrics["labels"] = np.concatenate(metrics["labels"], axis=0)
        
        dataset_size, num_samples, num_classes = metrics["probs"].shape
        bma_preds = metrics["probs"].mean(axis=1).argmax(axis=-1)
        metrics["bma_acc"] = np.mean([(bma_preds[metrics["labels"] == c] == c).mean() for c in range(num_classes)])
        probs = metrics["probs"][np.arange(dataset_size),:,metrics["labels"]]
        log_probs = np.sum(np.log(probs), axis=1)
        metrics["clml"] = scipy.special.logsumexp(log_probs) - np.log(len(log_probs))
    
    return metrics
