import os
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from addict import Dict
from typing import Any, List

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from ..utils.logger import setup_logger
from ..utils.metrices import AverageMeter

def run(config: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
) -> Any:
    """
    Run the training pipeline.

    Args:
        config (Dict): Configuration dictionary.
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        Any: Training results and metrics.
    """
    device = torch.device(config.device)
    model.to(device)
    
    torch.manual_seed(config.seed)

    os.makedirs(config.training.checkpoint.dir, exist_ok=True)
    os.makedirs(config.training.logging.dir, exist_ok=True)
    
    logger = setup_logger(config)

    # Setup optimizer, criterion, scheduler
    criterion = getattr(nn, config.training.criterion)()
    optimizer = getattr(torch.optim, config.training.optimizer.name)(
        model.parameters(), **config.training.optimizer.params
    )
    
    scheduler = None
    if config.training.lr_scheduler.name:
        scheduler = getattr(torch.optim.lr_scheduler, config.training.lr_scheduler.name)(
            optimizer, **config.training.lr_scheduler.params
        )

    # TODO: Initialize log writer (TensorBoard or WandB)
    # writer = SummaryWriter(log_dir=config.training.logging.dir) if config.training.logging.report_to == 'tensorboard' else None

    best_metric_val = -1.0
    
    logger.info("="*50)
    logger.info(f"Starting training for project: {config.training.logging.project_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Total epochs: {config.training.epochs}")
    logger.info(f"Optimizer: {config.training.optimizer.name}")
    if scheduler:
        logger.info(f"Scheduler: {config.training.lr_scheduler.name}")
    logger.info("="*50)

    for epoch in range(1, config.training.epochs + 1):
        
        # Train
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Val
        val_results = _validate_one_epoch(
            model, val_loader, criterion, device, config.training.metrics
        )
        
        # Logging
        if epoch % config.training.logging.log_freq == 0 or epoch == config.training.epochs:
            log_str = f"Epoch [{epoch}/{config.training.epochs}] | Train Loss: {train_loss:.4f}"
            for metric, value in val_results.items():
                log_str += f" | Val {metric}: {value:.4f}"
            logger.info(log_str)
        
            # TODO: Report to TensorBoard/WandB
            # if writer:
            #     writer.add_scalar('Loss/train', train_loss, epoch)
            #     for metric, value in val_results.items():
            #         writer.add_scalar(f'Metric/{metric}', value, epoch)

        # Update LR
        if scheduler:
            scheduler.step()

        # Save checkpoint
        current_metric_val = val_results[config.training.checkpoint.monitor]

        # save_strategy is 'best'
        if config.training.checkpoint.save_strategy == 'best':
            if current_metric_val > best_metric_val:
                best_metric_val = current_metric_val
                save_path = os.path.join(config.training.checkpoint.dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Best model saved to {save_path} with {config.training.checkpoint.monitor}: {best_metric_val:.4f}")
        
        # save_strategy is 'last'
        elif config.training.checkpoint.save_strategy == 'last':
            save_path = os.path.join(config.training.checkpoint.dir, "last_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Last model saved to {save_path}")

        # save_strategy is 'all' (by save_freq)
        elif config.training.checkpoint.save_strategy == 'epoch':
            if epoch % config.training.checkpoint.save_freq == 0:
                save_path = os.path.join(config.training.checkpoint.dir, f"epoch_{epoch}.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Model at epoch {epoch} saved to {save_path}")
                
    # if writer: writer.close()
    logger.info("="*50)
    logger.info("Training finished!")
    logger.info(f"Best validation metric ({config.training.checkpoint.monitor}): {best_metric_val:.4f}")
    logger.info("="*50)

    logging.shutdown()

    return {"best_metric": best_metric_val}

def _train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: Optimizer, 
    device: torch.device, 
) -> float:
    model.train()
    loss_meter = AverageMeter()
    progress_bar = tqdm(loader, desc="Training")

    for i, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), data.size(0))
        
        # Update progress bar
        if (i + 1) % 100 == 0:
            progress_bar.set_postfix(loss=loss_meter.avg)

    return loss_meter.avg

def _validate_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    metrics: List[str]
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Validating"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), data.size(0))
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    results = {"loss": loss_meter.avg}
    for metric in metrics:
        if metric == "acc":
            results[metric] = accuracy_score(all_labels, all_preds)
        elif metric == "balanced_acc":
            results[metric] = balanced_accuracy_score(all_labels, all_preds)
        elif metric == "f1":
            results[metric] = f1_score(all_labels, all_preds, average='weighted') # 'weighted' for multi-class
    
    return results


