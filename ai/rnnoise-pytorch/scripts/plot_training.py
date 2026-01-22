"""
Plot Training Results - Generate charts from JSON logs

Creates visualization from training_logger JSON outputs:
- Loss curves (total, gain, VAD)
- Learning rate schedule
- Comparison charts
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from datetime import datetime


def load_metrics(metrics_file):
    """Load metrics JSON file."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    return data


def plot_loss_curves(metrics_data, output_dir):
    """Plot training loss curves."""
    epochs = [ep['epoch'] for ep in metrics_data['epochs']]
    total_loss = [ep['train']['loss'] for ep in metrics_data['epochs']]
    gain_loss = [ep['train']['gain_loss'] for ep in metrics_data['epochs']]
    vad_loss = [ep['train']['vad_loss'] for ep in metrics_data['epochs']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Progress: {metrics_data['experiment']}", fontsize=16)
    
    # Plot 1: Total loss
    axes[0, 0].plot(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Component losses
    axes[0, 1].plot(epochs, gain_loss, 'r-', linewidth=2, label='Gain Loss')
    axes[0, 1].plot(epochs, vad_loss, 'g-', linewidth=2, label='VAD Loss (×1000)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Total loss (log scale)
    axes[1, 0].plot(epochs, total_loss, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Total Loss (Log Scale)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Loss smoothed (moving average)
    window = min(10, len(total_loss) // 10)
    if window > 1:
        smoothed = np.convolve(total_loss, np.ones(window)/window, mode='valid')
        smoothed_epochs = epochs[window-1:]
        axes[1, 1].plot(epochs, total_loss, 'b-', alpha=0.3, label='Raw')
        axes[1, 1].plot(smoothed_epochs, smoothed, 'b-', linewidth=2, label=f'Smoothed (w={window})')
    else:
        axes[1, 1].plot(epochs, total_loss, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Total Loss (Smoothed)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_learning_rate(metrics_data, output_dir):
    """Plot learning rate schedule."""
    epochs = [ep['epoch'] for ep in metrics_data['epochs']]
    lr = [ep['learning_rate'] for ep in metrics_data['epochs']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, lr, 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_convergence(metrics_data, output_dir):
    """Plot convergence analysis."""
    epochs = [ep['epoch'] for ep in metrics_data['epochs']]
    losses = [ep['train']['loss'] for ep in metrics_data['epochs']]
    
    # Compute improvement rate
    improvements = []
    for i in range(1, len(losses)):
        improvement = (losses[i-1] - losses[i]) / (losses[i-1] + 1e-8)
        improvements.append(improvement * 100)  # Percentage
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss with milestones
    axes[0].plot(epochs, losses, 'b-', linewidth=2)
    
    # Mark best epoch
    best_idx = np.argmin(losses)
    axes[0].plot(epochs[best_idx], losses[best_idx], 'r*', markersize=15, 
                label=f'Best: Epoch {epochs[best_idx]}')
    
    # Mark convergence (when improvement < 0.1%)
    if len(improvements) > 0:
        converged_idx = next((i for i, imp in enumerate(improvements) 
                            if abs(imp) < 0.1), len(improvements))
        if converged_idx < len(epochs) - 1:
            axes[0].axvline(x=epochs[converged_idx], color='orange', 
                          linestyle='--', label=f'Converged: Epoch {epochs[converged_idx]}')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Convergence')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Improvement rate
    if len(improvements) > 0:
        axes[1].plot(epochs[1:], improvements, 'g-', linewidth=2)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, 
                       label='Convergence threshold (0.1%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_title('Loss Improvement Rate')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'convergence.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(metrics_data, summary_data, output_dir):
    """Create summary statistics table."""
    epochs = [ep['epoch'] for ep in metrics_data['epochs']]
    losses = [ep['train']['loss'] for ep in metrics_data['epochs']]
    
    # Compute statistics
    stats = {
        'Total Epochs': len(epochs),
        'Best Epoch': int(np.argmin(losses) + 1),
        'Best Loss': f"{min(losses):.6f}",
        'Final Loss': f"{losses[-1]:.6f}",
        'Average Loss': f"{np.mean(losses):.6f}",
        'Loss Reduction': f"{(losses[0] - losses[-1]) / losses[0] * 100:.2f}%"
    }
    
    # Add from summary if available
    if summary_data and 'training_duration' in summary_data:
        start = datetime.fromisoformat(summary_data['training_duration']['start'])
        end = datetime.fromisoformat(summary_data['training_duration']['end'])
        duration = end - start
        hours = duration.total_seconds() / 3600
        stats['Training Time'] = f"{hours:.2f} hours"
    
    # Save as text
    output_path = os.path.join(output_dir, 'training_summary.txt')
    with open(output_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in stats.items():
            f.write(f"{key:20s}: {value}\n")
        
        f.write("\n" + "=" * 50 + "\n")
    
    print(f"Saved: {output_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate training visualization charts"
    )
    
    parser.add_argument('metrics_file', type=str,
                       help='Path to *_metrics.json file')
    parser.add_argument('output_dir', type=str,
                       help='Output directory for charts')
    
    parser.add_argument('--summary-file', type=str, default=None,
                       help='Optional *_summary.json file for additional info')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading metrics from: {args.metrics_file}")
    metrics_data = load_metrics(args.metrics_file)
    
    summary_data = None
    if args.summary_file and os.path.exists(args.summary_file):
        print(f"Loading summary from: {args.summary_file}")
        with open(args.summary_file, 'r') as f:
            summary_data = json.load(f)
    
    # Generate plots
    print("\nGenerating charts...")
    
    plot_loss_curves(metrics_data, args.output_dir)
    plot_learning_rate(metrics_data, args.output_dir)
    plot_convergence(metrics_data, args.output_dir)
    stats = create_summary_table(metrics_data, summary_data, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    print("=" * 50)
    
    print(f"\n✅ All charts saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
