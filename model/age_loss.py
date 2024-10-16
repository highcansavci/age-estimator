import torch
import torch.nn as nn

class AgeLoss(nn.Module):
    def __init__(self, lambda_uncertainty: float = 0.1):
        super().__init__()
        self.lambda_uncertainty = lambda_uncertainty

    def forward(self, pred_age: torch.Tensor, pred_uncertainty: torch.Tensor, 
                target_age: torch.Tensor) -> torch.Tensor:
        # Compute age prediction loss with uncertainty weighting
        age_loss = torch.abs(pred_age - target_age) * torch.exp(-pred_uncertainty)
        
        # Add uncertainty regularization term
        uncertainty_reg = self.lambda_uncertainty * pred_uncertainty
        
        # Combine losses
        total_loss = age_loss + uncertainty_reg
        return total_loss.mean()