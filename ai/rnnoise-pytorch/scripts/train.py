"""
RNNoise Training Script - Based on torch/rnnoise/train_rnnoise.py

Production training script with logging and better organization.
"""

import numpy as np
import torch
from torch import nn
import tqdm
import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rnnoise.model import RNNoise
from rnnoise.dataset import RNNoiseDataset
from rnnoise.loss import mask

# Optional: Import logger if available
try:
    from training_logger import TrainingLogger
    HAS_LOGGER = True
except:
    HAS_LOGGER = False
    print("Warning: training_logger not found, proceeding without JSON logging")


def create_parser():
    """Create argument parser matching reference implementation."""
    parser = argparse.ArgumentParser(
        description="Train RNNoise model for speech enhancement"
    )
    
    # Required arguments
    parser.add_argument('features', type=str, 
                       help='path to feature file in .f32 format')
    parser.add_argument('output', type=str, 
                       help='path to output folder')
    
    # Optional arguments
    parser.add_argument('--suffix', type=str, 
                       help="model name suffix", default="")
    parser.add_argument('--cuda-visible-devices', type=str, 
                       help="comma separated list of cuda visible device indices", 
                       default=None)
    
    # Model parameters
    model_group = parser.add_argument_group(title="model parameters")
    model_group.add_argument('--cond-size', type=int, 
                            help="first conditioning size, default: 128", 
                            default=128)
    model_group.add_argument('--gru-size', type=int, 
                            help="GRU hidden size, default: 384", 
                            default=384)
    
    # Training parameters
    training_group = parser.add_argument_group(title="training parameters")
    training_group.add_argument('--batch-size', type=int, 
                               help="batch size, default: 128", 
                               default=128)
    training_group.add_argument('--lr', type=float, 
                               help='learning rate, default: 1e-3', 
                               default=1e-3)
    training_group.add_argument('--epochs', type=int, 
                               help='number of training epochs, default: 200', 
                               default=200)
    training_group.add_argument('--sequence-length', type=int, 
                               help='sequence length, default: 2000', 
                               default=2000)
    training_group.add_argument('--lr-decay', type=float, 
                               help='learning rate decay factor, default: 5e-5', 
                               default=5e-5)
    training_group.add_argument('--initial-checkpoint', type=str, 
                               help='initial checkpoint to start training from', 
                               default=None)
    training_group.add_argument('--gamma', type=float, 
                               help='perceptual exponent (default 0.25)', 
                               default=0.25)
    training_group.add_argument('--sparse', action='store_true',
                               help='enable sparsification')
    
    # Logging
    logging_group = parser.add_argument_group(title="logging parameters")
    logging_group.add_argument('--log-dir', type=str,
                              help='directory for JSON logs',
                              default=None)
    logging_group.add_argument('--experiment-name', type=str,
                              help='experiment name for logging',
                              default='rnnoise')
    
    return parser


def main():
    """Main training function - exact match to reference."""
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup CUDA
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    
    # Constants from reference
    adam_betas = [0.8, 0.98]
    adam_eps = 1e-8
    
    # Create output directory
    checkpoint_dir = os.path.join(args.output, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logger = None
    if HAS_LOGGER and args.log_dir:
        logger = TrainingLogger(
            log_dir=args.log_dir,
            experiment_name=args.experiment_name
        )
        logger.log_config({
            'model': {
                'cond_size': args.cond_size,
                'gru_size': args.gru_size,
            },
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'lr_decay': args.lr_decay,
                'sequence_length': args.sequence_length,
                'gamma': args.gamma,
                'sparse': args.sparse,
            },
            'data': {
                'features_file': args.features,
            }
        })
    
    # Device selection
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize checkpoint dict (matching reference exactly)
    checkpoint = dict()
    checkpoint['model_args'] = ()
    checkpoint['model_kwargs'] = {
        'cond_size': args.cond_size, 
        'gru_size': args.gru_size
    }
    
    # Create model
    model = RNNoise(*checkpoint['model_args'], **checkpoint['model_kwargs'])
    
    # Load initial checkpoint if provided
    if args.initial_checkpoint is not None:
        print(f"Loading initial checkpoint from {args.initial_checkpoint}")
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    checkpoint['state_dict'] = model.state_dict()
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.features}")
    dataset = RNNoiseDataset(args.features, sequence_length=args.sequence_length)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=4
    )
    print(f"Dataset: {len(dataset)} sequences")
    
    # Optimizer (exact match to reference)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=adam_betas, 
        eps=adam_eps
    )
    
    # Learning rate scheduler (exact match to reference)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, 
        lr_lambda=lambda x: 1 / (1 + args.lr_decay * x)
    )
    
    # Move model to device
    model.to(device)
    
    # Training loop (exact match to reference)
    gamma = args.gamma
    states = None
    
    for epoch in range(1, args.epochs + 1):
        running_gain_loss = 0
        running_vad_loss = 0
        running_loss = 0
        
        print(f"training epoch {epoch}...")
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, gain, vad) in enumerate(tepoch):
                optimizer.zero_grad()
                
                # Move to device
                features = features.to(device)
                gain = gain.to(device)
                vad = vad.to(device)
                
                # Forward pass
                pred_gain, pred_vad, states = model(features, states=states)
                states = [state.detach() for state in states]
                
                # Trim predictions and targets (skip first 3 and last frame)
                gain = gain[:, 3:-1, :]
                vad = vad[:, 3:-1, :]
                
                # Target gain processing (exact match to reference)
                target_gain = torch.clamp(gain, min=0)
                target_gain = target_gain * (torch.tanh(8 * target_gain) ** 2)
                
                # Loss computation (exact match to reference)
                e = pred_gain ** gamma - target_gain ** gamma
                gain_loss = torch.mean((1 + 5.0 * vad) * mask(gain) * (e ** 2))
                
                # VAD loss (exact match to reference - binary cross-entropy)
                vad_loss = torch.mean(
                    torch.abs(2 * vad - 1) * (
                        -vad * torch.log(0.01 + pred_vad) - 
                        (1 - vad) * torch.log(1.01 - pred_vad)
                    )
                )
                
                # Total loss
                loss = gain_loss + 0.001 * vad_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Sparsification step (if enabled)
                if args.sparse:
                    model.sparsify()
                
                # Learning rate scheduling
                scheduler.step()
                
                # Accumulate losses
                running_gain_loss += gain_loss.detach().cpu().item()
                running_vad_loss += vad_loss.detach().cpu().item()
                running_loss += loss.detach().cpu().item()
                
                # Update progress bar
                tepoch.set_postfix(
                    loss=f"{running_loss/(i+1):8.5f}",
                    gain_loss=f"{running_gain_loss/(i+1):8.5f}",
                    vad_loss=f"{running_vad_loss/(i+1):8.5f}",
                )
        
        # Epoch statistics
        avg_loss = running_loss / len(dataloader)
        avg_gain_loss = running_gain_loss / len(dataloader)
        avg_vad_loss = running_vad_loss / len(dataloader)
        
        # Log to JSON if logger available
        if logger:
            logger.log_epoch(
                epoch=epoch,
                train_loss=avg_loss,
                train_gain_loss=avg_gain_loss,
                train_vad_loss=avg_vad_loss,
                learning_rate=scheduler.get_last_lr()[0]
            )
        
        # Save checkpoint (exact match to reference)
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'rnnoise{args.suffix}_{epoch}.pth'
        )
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = avg_loss
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final summary if logger available
    if logger:
        # Find best epoch
        best_epoch = 1
        best_loss = float('inf')
        for ep_data in logger.epoch_metrics:
            if ep_data['train']['loss'] < best_loss:
                best_loss = ep_data['train']['loss']
                best_epoch = ep_data['epoch']
        
        logger.save_summary(
            total_epochs=args.epochs,
            best_epoch=best_epoch,
            best_loss=best_loss,
            final_model_path=checkpoint_path
        )
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
