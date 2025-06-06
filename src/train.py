import os
import time
import json
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from src.logging import setup_logger, log_hyperparameters, log_metrics

from src.models.cnn.basic_cnns.basic_cnn import BasicCNN
# from src.models.cnn.basic_cnns.deeper_cnn import DeeperCNN
# from src.models.cnn.basic_cnns.vgg_mini import VGGMini
# from src.models.cnn.densenets.densenet121 import DenseNet121
# from src.models.cnn.densenets.densenet169 import DenseNet169
# from src.models.cnn.densenets.densenet_bc import DenseNetBC
# from src.models.cnn.efficientnets.efficientnet_b0 import EfficientNetB0
# from src.models.cnn.efficientnets.efficientnet_b1 import EfficientNetB1
# from src.models.cnn.efficientnets.efficientnet_b2 import EfficientNetB2
# from src.models.cnn.resnets.resnet18 import ResNet18
# from src.models.cnn.resnets.resnet34 import ResNet34
# from src.models.cnn.resnets.resnet50 import ResNet50

# from src.models.hybrid.attn_enh_cnn.botnet50 import BotNet50
# from src.models.hybrid.attn_enh_cnn.lambda_resnet import LambdaResNet
# from src.models.hybrid.cnn_tfmr_hybrid.cait_xxs24 import CaiTXXS24
# from src.models.hybrid.cnn_tfmr_hybrid.cvt_13 import CvT13
# from src.models.hybrid.cnn_tfmr_hybrid.pit_tiny import PiTTiny
# from src.models.hybrid.convnext.convnext_tiny import ConvNeXtTiny
# from src.models.hybrid.convnext.convnext_small import ConvNeXtSmall
# from src.models.hybrid.convnext.convnext_atto import ConvNeXtAtto

# from src.models.lightweight.mobilenet.mobilenet_v2 import MobileNetv2
# from src.models.lightweight.mobilenet.mobilenet_v3_small import MobileNetv3Small
# from src.models.lightweight.mobilenet.mobilenet_v3_large import MobileNetv3Large
# from src.models.lightweight.shufflenet.shufflenet_v1 import ShuffleNetv1
# from src.models.lightweight.shufflenet.shufflenet_v2 import ShuffleNetv2
# from src.models.lightweight.ultra_lw.ghostnet import GhostNet
# from src.models.lightweight.ultra_lw.micronet import MicroNet
# from src.models.lightweight.ultra_lw.squeezenet import SqueezeNet

# from src.models.transformer.compact.cct_7 import CCT7
# from src.models.transformer.compact.cct_14 import CCT14
# from src.models.transformer.deit.deit_tiny import DeITTiny
# from src.models.transformer.deit.deit_small import DeiTSmall
# from src.models.transformer.deit.deit_distilled import DeiTDistilled
# from src.models.transformer.swin.swin_tiny import SwinTiny
# from src.models.transformer.swin.swin_small import SwinSmall
# from src.models.transformer.vit.vit_tiny import ViTTiny
# from src.models.transformer.vit.vit_small import ViTSmall
# from src.models.transformer.vit.vit_base import ViTBase
# from src.models.transformer.vit.vit_patch4 import ViTPatch4

def build_model(config):
    model_params = config['model']
    if model_params.get('type', '').lower() == 'basiccnn':
        module = importlib.import_module("src.models.cnn.basic_cnns.basic_cnn")
        model_class = getattr(module, "BasicCNN")
        model = model_class(
            num_classes=model_params.get('num_classes', 10),
            conv1_channels=model_params.get('conv1_channels', 32),
            conv2_channels=model_params.get('conv2_channels', 64),
            conv3_channels=model_params.get('conv3_channels', 128),
            kernel_size=model_params.get('kernel_size', 3),
            dropout_rate=model_params.get('dropout', 0.25),
            fc1_size=model_params.get('fc1_size', 256),
            fc2_size=model_params.get('fc2_size', 128)
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {model_params.get('type')}")

def get_data_loaders(config, fold=0):
    """Set up CIFAR-10 data loaders, optionally with K-fold split."""
    training_config = config['training']
    batch_size = training_config.get('batch_size', 64)
    k_folds = training_config.get('k_folds', 5)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.247, 0.243, 0.261))
    ])

    dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    if k_folds <= 1:
        # No cross-validation, use all data for training and validation
        train_indices = list(range(0, int(0.9 * len(dataset))))
        val_indices = list(range(int(0.9 * len(dataset)), len(dataset)))
    else:
        # Simple K-fold split
        fold_size = len(dataset) // k_folds
        indices = list(range(len(dataset)))
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    avg_loss = running_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    if total == 0:
        print("WARNING: Validation loader returned zero samples.")
        return 0.0, 0.0
    avg_loss = running_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

def run_training(config_path, experiment_name, fold=0):
    # Read config
    with open(config_path, 'r') as f:
        config = json.load(f)
    training_config = config['training']
    num_epochs = training_config.get('num_epochs', 20)
    lr = training_config.get('learning_rate', 0.001)
    k_folds = training_config.get('k_folds', 5)

    # Set up logging
    log_dir = Path('logs') / experiment_name
    logger = setup_logger(
        name=f"train_fold{fold}",
        log_dir=str(log_dir),
        console_output=True
    )
    
    logger.info(f"Starting training for fold {fold}")
    logger.info(f"Using device: {'mps' if torch.mps.is_available() else 'cpu'}")
    
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    # Log hyperparameters
    log_hyperparameters(logger, {
        'learning_rate': lr,
        'num_epochs': num_epochs,
        'k_folds': k_folds,
        'fold': fold,
        **config
    })

    # Data loaders
    train_loader, val_loader = get_data_loaders(config, fold=fold)
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

    # Model, loss, optimizer
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    save_dir = Path('weights') / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"model_fold{fold}.pth"

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        log_metrics(logger, {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch_time': epoch_time
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved at epoch {epoch} with validation accuracy: {val_acc:.2f}%")

    total_time = time.time() - start_time
    
    # Log final results
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    logger.info(f"Best training accuracy: {best_train_acc:.2f}%")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Model saved to: {model_path}")

    results = {
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_time": total_time,
        "model_path": str(model_path)
    }
    return results
