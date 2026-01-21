"""
Basic RNNoise Training Example
"""

import torch
from rnnoise.model import RNNoise
from rnnoise.dataset import RNNoiseDataset
from rnnoise.loss import rnnoise_loss

# 1. Tạo model
model = RNNoise(
    input_dim=42,
    output_dim=22,
    cond_size=128,
    gru_size=384  # 384 = best quality, 256 = faster
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. Load dataset
dataset = RNNoiseDataset(
    features_file="features.f32",
    sequence_length=2000
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

# 3. Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=[0.8, 0.98],
    eps=1e-8
)

# 4. Training loop (simplified)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 11):  # 10 epochs for demo
    running_loss = 0.0
    
    for i, (features, gains, vad) in enumerate(dataloader):
        # Move to device
        features = features.to(device)
        gains = gains.to(device)
        vad = vad.to(device)
        
        # Forward pass
        pred_gains, pred_vad, _ = model(features)
        
        # Compute loss
        losses = rnnoise_loss(
            pred_gains[:, 3:-1, :],  # Skip first/last frames
            pred_vad[:, 3:-1, :],
            gains[:, 3:-1, :],
            vad[:, 3:-1, :],
            gamma=0.25
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Sparsify (if enabled)
        # model.sparsify()
        
        running_loss += losses['total'].item()
    
    # Print epoch stats
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f"checkpoint_epoch_{epoch}.pth")

print("\n✅ Training complete!")
