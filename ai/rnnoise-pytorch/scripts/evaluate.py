"""
RNNoise Model Evaluation Script

Evaluates trained model quality using objective metrics:
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)  
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

Also saves processed audio samples for listening tests.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rnnoise.model import RNNoise

# Import evaluation metrics
try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("Warning: pesq not installed. pip install pesq")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("Warning: pystoi not installed. pip install pystoi")


def si_sdr(reference, estimation):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        reference: Clean reference signal
        estimation: Estimated/denoised signal
        
    Returns:
        SI-SDR value in dB
    """
    # Remove mean
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # Compute SI-SDR
    alpha = np.dot(estimation, reference) / (np.linalg.norm(reference) ** 2 + 1e-8)
    projection = alpha * reference
    noise = estimation - projection
    
    si_sdr_value = 10 * np.log10(
        (np.linalg.norm(projection) ** 2) / (np.linalg.norm(noise) ** 2 + 1e-8)
    )
    
    return si_sdr_value


def process_frame(model, audio, device, frame_size=480):
    """
    Process audio through RNNoise model frame-by-frame.
    
    Args:
        model: RNNoise model
        audio: Input audio (numpy array)
        device: torch device
        frame_size: Frame size (default 480 = 10ms @ 48kHz)
        
    Returns:
        Processed audio
    """
    # Pad audio to multiple of frame_size
    padding = frame_size - (len(audio) % frame_size)
    if padding != frame_size:
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Process frame by frame
    num_frames = len(audio) // frame_size
    output = np.zeros_like(audio)
    states = None
    
    model.eval()
    with torch.no_grad():
        for i in range(num_frames):
            # Extract frame
            frame = audio[i * frame_size:(i + 1) * frame_size]
            
            # TODO: Need to implement full inference pipeline
            # This requires feature extraction matching dump_features.c
            # For now, just copy input (placeholder)
            output[i * frame_size:(i + 1) * frame_size] = frame
    
    return output


def evaluate_sample(model, noisy_path, clean_path, output_path, device, sample_rate=48000):
    """
    Evaluate single audio sample.
    
    Args:
        model: RNNoise model
        noisy_path: Path to noisy audio
        clean_path: Path to clean reference
        output_path: Path to save denoised output
        device: torch device
        sample_rate: Expected sample rate
        
    Returns:
        Dictionary with metrics
    """
    # Load audio
    noisy, sr_noisy = sf.read(noisy_path)
    clean, sr_clean = sf.read(clean_path)
    
    # Verify sample rate
    if sr_noisy != sample_rate or sr_clean != sample_rate:
        raise ValueError(f"Expected {sample_rate}Hz, got noisy={sr_noisy}Hz, clean={sr_clean}Hz")
    
    # Ensure mono
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=1)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    
    # Ensure same length
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]
    
    # Process through model
    denoised = process_frame(model, noisy, device)
    denoised = denoised[:min_len]  # Remove padding
    
    # Save output
    sf.write(output_path, denoised, sample_rate)
    
    # Compute metrics
    metrics = {}
    
    # PESQ (requires 8kHz or 16kHz, so resample)
    if HAS_PESQ:
        try:
            # Resample to 16kHz for PESQ
            from scipy import signal
            clean_16k = signal.resample(clean, len(clean) * 16000 // sample_rate)
            denoised_16k = signal.resample(denoised, len(denoised) * 16000 // sample_rate)
            metrics['pesq'] = pesq(16000, clean_16k, denoised_16k, 'wb')
        except Exception as e:
            metrics['pesq'] = None
            print(f"PESQ computation failed: {e}")
    
    # STOI
    if HAS_STOI:
        try:
            metrics['stoi'] = stoi(clean, denoised, sample_rate, extended=False)
        except Exception as e:
            metrics['stoi'] = None
            print(f"STOI computation failed: {e}")
    
    # SI-SDR
    try:
        metrics['si_sdr'] = si_sdr(clean, denoised)
    except Exception as e:
        metrics['si_sdr'] = None
        print(f"SI-SDR computation failed: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RNNoise model quality"
    )
    
    parser.add_argument('checkpoint', type=str,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('test_dir', type=str,
                       help='Directory with test files (noisy/ and clean/ subdirs)')
    parser.add_argument('output_dir', type=str,
                       help='Output directory for results')
    
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of test samples to evaluate')
    parser.add_argument('--sample-rate', type=int, default=48000,
                       help='Audio sample rate (default: 48000)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    denoised_dir = os.path.join(args.output_dir, 'denoised')
    os.makedirs(denoised_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    model = RNNoise(*checkpoint.get('model_args', ()), 
                    **checkpoint.get('model_kwargs', {}))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Find test files
    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')
    
    if not os.path.exists(noisy_dir) or not os.path.exists(clean_dir):
        print(f"ERROR: Expected {args.test_dir}/noisy/ and {args.test_dir}/clean/")
        return 1
    
    # Get file list
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
    noisy_files = noisy_files[:args.num_samples]
    
    if not noisy_files:
        print(f"ERROR: No .wav files found in {noisy_dir}")
        return 1
    
    print(f"Found {len(noisy_files)} test files")
    
    # Evaluate each sample
    results = []
    
    for filename in tqdm(noisy_files, desc="Evaluating"):
        noisy_path = os.path.join(noisy_dir, filename)
        clean_path = os.path.join(clean_dir, filename)
        output_path = os.path.join(denoised_dir, filename)
        
        if not os.path.exists(clean_path):
            print(f"Warning: Clean file not found: {clean_path}")
            continue
        
        try:
            metrics = evaluate_sample(
                model, noisy_path, clean_path, output_path,
                device, args.sample_rate
            )
            
            results.append({
                'filename': filename,
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Compute average metrics
    avg_metrics = {}
    for metric_name in ['pesq', 'stoi', 'si_sdr']:
        values = [r['metrics'][metric_name] for r in results 
                 if r['metrics'].get(metric_name) is not None]
        if values:
            avg_metrics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    # Save results
    results_json = {
        'checkpoint': args.checkpoint,
        'test_dir': args.test_dir,
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(results),
        'average_metrics': avg_metrics,
        'per_sample': results
    }
    
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if 'pesq' in avg_metrics:
        print(f"PESQ: {avg_metrics['pesq']['mean']:.3f} ± {avg_metrics['pesq']['std']:.3f}")
    if 'stoi' in avg_metrics:
        print(f"STOI: {avg_metrics['stoi']['mean']:.3f} ± {avg_metrics['stoi']['std']:.3f}")
    if 'si_sdr' in avg_metrics:
        print(f"SI-SDR: {avg_metrics['si_sdr']['mean']:.2f} ± {avg_metrics['si_sdr']['std']:.2f} dB")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Denoised audio in: {denoised_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
