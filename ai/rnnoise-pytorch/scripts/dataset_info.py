"""
Dataset Information Tracker

Documents dataset used for training:
- Sources and durations
- SNR distribution
- Statistics
- Metadata for reproducibility
"""

import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
from datetime import datetime
from tqdm import tqdm


def analyze_audio_file(filepath):
    """
    Analyze single audio file.
    
    Returns:
        Dictionary with file statistics
    """
    try:
        data, samplerate = sf.read(filepath)
        
        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        duration = len(data) / samplerate
        rms = np.sqrt(np.mean(data ** 2))
        peak = np.max(np.abs(data))
        
        return {
            'filename': os.path.basename(filepath),
            'duration_sec': float(duration),
            'sample_rate': int(samplerate),
            'num_samples': int(len(data)),
            'rms_level': float(rms),
            'peak_level': float(peak),
            'rms_db': float(20 * np.log10(rms + 1e-8)),
            'peak_db': float(20 * np.log10(peak + 1e-8))
        }
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'error': str(e)
        }


def analyze_directory(directory, file_pattern='*.wav'):
    """
    Analyze all audio files in directory.
    
    Returns:
        List of file statistics
    """
    import glob
    
    files = glob.glob(os.path.join(directory, file_pattern))
    
    if not files:
        return []
    
    print(f"Analyzing {len(files)} files in {directory}...")
    
    results = []
    for filepath in tqdm(files):
        info = analyze_audio_file(filepath)
        results.append(info)
    
    return results


def compute_summary_stats(file_stats):
    """
    Compute summary statistics from file list.
    
    Returns:
        Dictionary with aggregate stats
    """
    # Filter out errors
    valid_stats = [s for s in file_stats if 'error' not in s]
    
    if not valid_stats:
        return {}
    
    durations = [s['duration_sec'] for s in valid_stats]
    rms_levels = [s['rms_db'] for s in valid_stats]
    
    summary = {
        'num_files': len(valid_stats),
        'total_duration_sec': float(np.sum(durations)),
        'total_duration_hours': float(np.sum(durations) / 3600),
        'avg_duration_sec': float(np.mean(durations)),
        'min_duration_sec': float(np.min(durations)),
        'max_duration_sec': float(np.max(durations)),
        'avg_rms_db': float(np.mean(rms_levels)),
        'std_rms_db': float(np.std(rms_levels)),
        'sample_rates': list(set(s['sample_rate'] for s in valid_stats)),
        'num_errors': len(file_stats) - len(valid_stats)
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Document dataset used for RNNoise training"
    )
    
    parser.add_argument('--clean-speech', type=str, required=True,
                       help='Directory with clean speech files')
    parser.add_argument('--background-noise', type=str, required=True,
                       help='Directory with background noise files')
    parser.add_argument('--foreground-noise', type=str, default=None,
                       help='Directory with foreground noise files (optional)')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for dataset info')
    
    parser.add_argument('--description', type=str, default='',
                       help='Description of this dataset')
    parser.add_argument('--language', type=str, default='',
                       help='Primary language (e.g., Vietnamese)')
    
    args = parser.parse_args()
    
    # Analyze each category
    dataset_info = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'description': args.description,
            'language': args.language
        },
        'sources': {}
    }
    
    # Clean speech
    print("\n=== Clean Speech ===")
    clean_stats = analyze_directory(args.clean_speech)
    clean_summary = compute_summary_stats(clean_stats)
    
    dataset_info['sources']['clean_speech'] = {
        'directory': args.clean_speech,
        'summary': clean_summary,
        'files': clean_stats[:10]  # Save only first 10 to keep JSON manageable
    }
    
    print(f"Total: {clean_summary.get('total_duration_hours', 0):.2f} hours")
    
    # Background noise
    print("\n=== Background Noise ===")
    bg_stats = analyze_directory(args.background_noise)
    bg_summary = compute_summary_stats(bg_stats)
    
    dataset_info['sources']['background_noise'] = {
        'directory': args.background_noise,
        'summary': bg_summary,
        'files': bg_stats[:10]
    }
    
    print(f"Total: {bg_summary.get('total_duration_hours', 0):.2f} hours")
    
    # Foreground noise (optional)
    if args.foreground_noise:
        print("\n=== Foreground Noise ===")
        fg_stats = analyze_directory(args.foreground_noise)
        fg_summary = compute_summary_stats(fg_stats)
        
        dataset_info['sources']['foreground_noise'] = {
            'directory': args.foreground_noise,
            'summary': fg_summary,
            'files': fg_stats[:10]
        }
        
        print(f"Total: {fg_summary.get('total_duration_hours', 0):.2f} hours")
    
    # Compute overall statistics
    total_clean_hours = clean_summary.get('total_duration_hours', 0)
    total_noise_hours = bg_summary.get('total_duration_hours', 0)
    
    dataset_info['overall'] = {
        'total_clean_hours': total_clean_hours,
        'total_noise_hours': total_noise_hours,
        'clean_to_noise_ratio': total_clean_hours / (total_noise_hours + 1e-8)
    }
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Clean Speech:      {total_clean_hours:.2f} hours ({clean_summary.get('num_files', 0)} files)")
    print(f"Background Noise:  {total_noise_hours:.2f} hours ({bg_summary.get('num_files', 0)} files)")
    
    if args.foreground_noise:
        fg_hours = fg_summary.get('total_duration_hours', 0)
        print(f"Foreground Noise:  {fg_hours:.2f} hours ({fg_summary.get('num_files', 0)} files)")
    
    print(f"\nLanguage: {args.language if args.language else 'Not specified'}")
    print(f"Description: {args.description if args.description else 'None'}")
    print("=" * 60)
    
    print(f"\n✅ Dataset info saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
