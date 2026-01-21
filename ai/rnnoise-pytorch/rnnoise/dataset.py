"""
RNNoise Dataset - Load .f32 feature files for training
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RNNoiseDataset(Dataset):
    """
    Dataset cho RNNoise training.
    
    Load features từ file .f32 generated bởi dump_features tool.
    Format: [features(42), gains(32), vad(1)] × sequences
    
    Args:
        features_file: Path to .f32 feature file  
        sequence_length: Number of frames per sequence (default: 2000 = 20s)
    """
    
    def __init__(self, features_file: str, sequence_length: int = 2000):
        self.sequence_length = sequence_length
        
        # Memory-mapped file (efficient cho large files)
        self.data = np.memmap(features_file, dtype='float32', mode='r')
        
        # Feature dimension: 42 + 32 + 1 = 75 (old) or 65 + 32 + 1 = 98 (new)
        dim = 98  # Updated dimension
        
        # Calculate number of complete sequences
        self.nb_sequences = self.data.shape[0] // self.sequence_length // dim
        
        # Trim to multiple of sequence_length
        self.data = self.data[:self.nb_sequences * sequence_length * dim]
        
        # Reshape: [num_sequences, sequence_length, features]
        self.data = np.reshape(
            self.data, 
            (self.nb_sequences, self.sequence_length, dim)
        )
    
    def __len__(self) -> int:
        return self.nb_sequences
    
    def __getitem__(self, index: int):
        """
        Returns:
            features: [sequence_length, 65] - Input features (42 + padding)
            gains: [sequence_length, 32] - Target gains 
            vad: [sequence_length, 1] - Voice activity detection target
        """
        # Split dimensions
        features = self.data[index, :, :65].copy()  # Input features
        gains = self.data[index, :, 65:-1].copy()   # Target gains
        vad = self.data[index, :, -1:].copy()       # VAD target
        
        return features, gains, vad


# === Example Usage ===
if __name__ == "__main__":
    # Load dataset
    dataset = RNNoiseDataset("features.f32", sequence_length=2000)
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Sequence length: {dataset.sequence_length} frames (20 seconds)")
    
    # Get one sample
    features, gains, vad = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Features: {features.shape}")  # [2000, 65]
    print(f"  Gains: {gains.shape}")        # [2000, 32] 
    print(f"  VAD: {vad.shape}")            # [2000, 1]
