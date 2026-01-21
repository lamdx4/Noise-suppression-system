"""
RNNoise Loss Functions - Perceptual losses for training
"""

import torch
import torch.nn.functional as F


def mask(g: torch.Tensor) -> torch.Tensor:
    """
    Mask cho invalid gains (những gain = -1 trong ground truth).
    
    Args:
        g: Gain tensor
        
    Returns:
        Mask tensor (1 for valid, 0 for invalid)
    """
    return torch.clamp(g + 1, max=1)


def perceptual_gain_loss(
    pred_gain: torch.Tensor,
    target_gain: torch.Tensor,
    vad: torch.Tensor,
    gamma: float = 0.25
) -> torch.Tensor:
    """
    Perceptual gain loss - Custom loss cho dự đoán gains.
    
    Sử dụng power transform (gamma) để tăng penalty cho errors ở low gains.
    Weighted by VAD để focus vào speech frames.
    
    Args:
        pred_gain: Predicted gains [B, T, 22]
        target_gain: Target gains [B, T, 22]
        vad: Voice activity [B, T, 1]
        gamma: Perceptual exponent (default: 0.25)
        
    Returns:
        Scalar loss value
    """
    # Clamp target gains to valid range [0, 1]
    target_gain = torch.clamp(target_gain, min=0)
    
    # Apply tanh squashing (smooth threshold)
    target_gain = target_gain * (torch.tanh(8 * target_gain) ** 2)
    
    # Perceptual error (power domain)
    error = pred_gain ** gamma - target_gain ** gamma
    
    # Weight by VAD (speech frames 6× more important)
    vad_weight = 1 + 5.0 * vad
    
    # Apply mask (ignore invalid targets)
    gain_mask = mask(target_gain)
    
    # Weighted MSE
    loss = torch.mean(vad_weight * gain_mask * (error ** 2))
    
    return loss


def vad_loss(
    pred_vad: torch.Tensor,
    target_vad: torch.Tensor
) -> torch.Tensor:
    """
    VAD loss - Binary cross-entropy weighted by confidence.
    
    Args:
        pred_vad: Predicted VAD probability [B, T, 1]
        target_vad: Target VAD [B, T, 1]
        
    Returns:
        Scalar loss value
    """
    # Confidence weighting (high weight at extremes 0/1, low at 0.5)
    confidence_weight = torch.abs(2 * target_vad - 1)
    
    # Binary cross-entropy (numerical stability với epsilon)
    bce = -(
        target_vad * torch.log(pred_vad + 0.01) +
        (1 - target_vad) * torch.log(1.01 - pred_vad)
    )
    
    # Weighted BCE
    loss = torch.mean(confidence_weight * bce)
    
    return loss


def rnnoise_loss(
    pred_gain: torch.Tensor,
    pred_vad: torch.Tensor,
    target_gain: torch.Tensor,
    target_vad: torch.Tensor,
    gamma: float = 0.25,
    vad_weight: float = 0.001
) -> dict:
    """
    Combined RNNoise loss.
    
    Args:
        pred_gain: Predicted gains
        pred_vad: Predicted VAD
        target_gain: Target gains
        target_vad: Target VAD
        gamma: Perceptual exponent (default: 0.25)
        vad_weight: Weight for VAD loss (default: 0.001)
        
    Returns:
        Dictionary with 'total', 'gain_loss', 'vad_loss'
    """
    # Individual losses
    gain_loss_val = perceptual_gain_loss(
        pred_gain, target_gain, target_vad, gamma
    )
    vad_loss_val = vad_loss(pred_vad, target_vad)
    
    # Combined (VAD là auxiliary task - weight nhỏ hơn)
    total_loss = gain_loss_val + vad_weight * vad_loss_val
    
    return {
        'total': total_loss,
        'gain_loss': gain_loss_val,
        'vad_loss': vad_loss_val
    }


# === Example Usage ===
if __name__ == "__main__":
    # Dummy data
    batch_size, seq_len, num_bands = 4, 100, 22
    
    pred_gain = torch.rand(batch_size, seq_len, num_bands)
    target_gain = torch.rand(batch_size, seq_len, num_bands)
    pred_vad = torch.rand(batch_size, seq_len, 1)
    target_vad = (torch.rand(batch_size, seq_len, 1) > 0.5).float()
    
    # Compute loss
    losses = rnnoise_loss(pred_gain, pred_vad, target_gain, target_vad)
    
    print("Losses:")
    print(f"  Total: {losses['total'].item():.6f}")
    print(f"  Gain: {losses['gain_loss'].item():.6f}")
    print(f"  VAD: {losses['vad_loss'].item():.6f}")
