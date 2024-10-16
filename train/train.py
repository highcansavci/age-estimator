import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import wandb
from scipy.stats import pearsonr
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.age_dataset import AgeDataset
from config.config import *
from model.age_estimator import AgeEstimator
from model.age_loss import AgeLoss

class AdvancedAugmentation:
    """Advanced augmentation techniques for age estimation"""
    
    def __init__(self, img_size: int):
        self.train_transform = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
                A.HueSaturationValue(p=0.8),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.test_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class AgeEstimationMetrics:
    """Custom metrics for age estimation"""
    
    @staticmethod
    def calculate_metrics(predictions: torch.Tensor, uncertainties: torch.Tensor, 
                         targets: torch.Tensor) -> Dict[str, float]:
        predictions = predictions.cpu().numpy()
        uncertainties = uncertainties.cpu().numpy()
        targets = targets.cpu().numpy()
        
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        acc_1 = np.mean(np.abs(predictions - targets) <= 1) * 100
        acc_3 = np.mean(np.abs(predictions - targets) <= 3) * 100
        acc_5 = np.mean(np.abs(predictions - targets) <= 5) * 100
        
        correlation, _ = pearsonr(predictions, targets)
        
        weighted_mae = np.mean(np.abs(predictions - targets) * np.exp(-uncertainties))
        mean_uncertainty = np.mean(uncertainties)
        
        error_within_uncertainty = np.mean(
            np.abs(predictions - targets) <= uncertainties
        ) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'acc_1': acc_1,
            'acc_3': acc_3,
            'acc_5': acc_5,
            'correlation': correlation,
            'weighted_mae': weighted_mae,
            'mean_uncertainty': mean_uncertainty,
            'calibration_score': error_within_uncertainty
        }

class AdvancedAgeEstimationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: AgeLoss,
        device: torch.device,
        config: Dict,
        enable_wandb: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.config = config
        self.enable_wandb = enable_wandb
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.metrics = AgeEstimationMetrics()
        
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.current_epoch = 0
        
        if self.enable_wandb and not wandb.run:
            wandb.init(
                project="age-estimator",
                config=self.config,
                name=f"training-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            
        self.setup_advanced_training()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'] * 10,
            epochs=self.config['epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

    def setup_advanced_training(self):
        self.mixup_enabled = self.config.get('use_mixup', True)
        self.mixup_alpha = self.config.get('mixup_alpha', 0.2)
        
        if self.config.get('use_swa', True):
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer,
                swa_lr=self.config['learning_rate'],
                anneal_strategy="cos",
                anneal_epochs=self.config.get('swa_anneal_epochs', 5)
            )

    def _apply_mixup(self, face_images: torch.Tensor, full_images: torch.Tensor, 
                    ages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation to the batch"""
        if self.mixup_enabled and random.random() > 0.5:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = face_images.size()[0]
            index = torch.randperm(batch_size).to(self.device)
            
            mixed_face_images = lam * face_images + (1 - lam) * face_images[index]
            mixed_full_images = lam * full_images + (1 - lam) * full_images[index]
            mixed_ages = lam * ages + (1 - lam) * ages[index]
            
            return mixed_face_images, mixed_full_images, mixed_ages
        return face_images, full_images, ages

    def _save_model(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'epoch': self.current_epoch,
            'best_loss': self.best_loss
        }
        
        if hasattr(self, 'swa_model'):
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
            
        torch.save(checkpoint, filename)

    def _save_checkpoint(self, epoch: int, train_loss: float, test_loss: float):
        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
        self._save_model(checkpoint_filename)
        
        if self.enable_wandb:
            wandb.save(checkpoint_filename)

    def _log_iteration_metrics(self, iteration: int, loss: float, lr: float):
        logging.info(
            f"Iteration {iteration}: Loss = {loss:.4f}, LR = {lr:.6f}"
        )
        
        if self.enable_wandb:
            wandb.log({
                'iteration_loss': loss,
                'learning_rate': lr,
                'iteration': iteration
            })

    def _log_epoch_metrics(self, epoch: int, train_loss: float, train_metrics: Dict,
                          test_loss: float, test_metrics: Dict):
        logging.info(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}"
        )
        
        if self.enable_wandb:
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss
            }
            
            # Add train metrics with 'train_' prefix
            metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
            # Add test metrics with 'test_' prefix
            metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
            
            wandb.log(metrics)

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        
        for i, (face_images, full_images, ages) in enumerate(self.train_loader):
            face_images = face_images.to(self.device)
            full_images = full_images.to(self.device)
            ages = ages.to(self.device).float()
            
            # Apply mixup augmentation
            face_images, full_images, ages = self._apply_mixup(face_images, full_images, ages)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_ages, uncertainties = self.model(face_images, full_images)
            loss = self.criterion(pred_ages, uncertainties, ages)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_value']
            )
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Collect predictions for metrics
            running_loss += loss.item() * face_images.size(0)
            all_predictions.extend(pred_ages.detach())
            all_uncertainties.extend(uncertainties.detach())
            all_targets.extend(ages.detach())
            
            # Log iteration metrics
            if (i + 1) % self.config['log_interval'] == 0:
                self._log_iteration_metrics(
                    i, 
                    loss.item(), 
                    self.optimizer.param_groups[0]['lr']
                )
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_metrics = self.metrics.calculate_metrics(
            torch.stack(all_predictions),
            torch.stack(all_uncertainties),
            torch.stack(all_targets)
        )
        
        # Update SWA if enabled
        if hasattr(self, 'swa_model') and self.current_epoch >= self.config.get('swa_start_epoch', 10):
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        
        return epoch_loss, epoch_metrics

    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        
        with torch.no_grad():
            for face_images, full_images, ages in self.test_loader:
                face_images = face_images.to(self.device)
                full_images = full_images.to(self.device)
                ages = ages.to(self.device).float()
                
                pred_ages, uncertainties = self.model(face_images, full_images)
                loss = self.criterion(pred_ages, uncertainties, ages)
                
                running_loss += loss.item() * face_images.size(0)
                all_predictions.extend(pred_ages)
                all_uncertainties.extend(uncertainties)
                all_targets.extend(ages)
        
        test_loss = running_loss / len(self.test_loader.dataset)
        test_metrics = self.metrics.calculate_metrics(
            torch.stack(all_predictions),
            torch.stack(all_uncertainties),
            torch.stack(all_targets)
        )
        
        return test_loss, test_metrics

    def train(self):
        """Main training loop"""
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train and evaluate
            train_loss, train_metrics = self.train_epoch()
            test_loss, test_metrics = self.evaluate()
            
            # Log epoch metrics
            self._log_epoch_metrics(epoch, train_loss, train_metrics, test_loss, test_metrics)
            
            # Save best model and check early stopping
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self._save_model('best_model.pth')
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
            if self.early_stopping_counter >= self.config['early_stopping_patience']:
                logging.info("Early stopping triggered")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self._save_checkpoint(epoch, train_loss, test_loss)
                
def main():
    config = {
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'epochs': 25,
        'batch_size': 4,
        'gradient_clip_value': 1.0,
        'early_stopping_patience': 5,
        'checkpoint_interval': 5,
        'log_interval': 1,
        'img_size': 224,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'use_swa': True,
        'swa_start_epoch': 10,
        'swa_anneal_epochs': 5,
        'enable_wandb': True,
        'lambda_uncertainty': 0.1
    }
    
    try:
        # Initialize augmentations
        augmentation = AdvancedAugmentation(config['img_size'])
        
        # Create datasets with proper augmentations
        train_dataset = AgeDataset(
            csv_file="../dataset/train.csv",
            root_dir="../UTKface_inthewild",
            transform=augmentation.train_transform,
            cache_dir="../cache_train",
            use_cache=True
        )
        
        test_dataset = AgeDataset(
            csv_file="../dataset/test.csv",
            root_dir="../UTKface_inthewild",
            transform=augmentation.test_transform,
            cache_dir="../cache_test",
            use_cache=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
        )
        
        # Initialize model, loss, and trainer
        model = AgeEstimator().to(DEVICE)
        criterion = AgeLoss(lambda_uncertainty=config['lambda_uncertainty'])
        
        trainer = AdvancedAgeEstimationTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            device=DEVICE,
            config=config,
            enable_wandb=config['enable_wandb']
        )
        
        # Start training
        trainer.train()

    finally:
        if config['enable_wandb'] and wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()