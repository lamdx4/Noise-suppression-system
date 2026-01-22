"""
Export RNNoise PyTorch Model to C - Based on dump_rnnoise_weights.py

This script exports a trained PyTorch model to C code for embedded deployment.
Uses weight-exchange library from reference implementation.
"""

import os
import sys
import argparse
import torch
from torch import nn

# Add paths
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, '../references/rnnoise/torch'))

# Try to import weight-exchange
try:
    sys.path.append(os.path.join(project_root, '../references/rnnoise/torch/weight-exchange'))
    import wexchange.torch
    import wexchange.c_export.c_writer
    HAS_WEXCHANGE = True
except ImportError:
    HAS_WEXCHANGE = False
    print("ERROR: weight-exchange library not found!")
    print("Make sure ai/references/rnnoise/torch/weight-exchange exists")
    sys.exit(1)

from rnnoise.model import RNNoise

# Layers that should NOT be quantized (from reference)
UNQUANTIZED_LAYERS = ['conv1', 'dense_out', 'vad_dense']

DESCRIPTION = f"""
Export RNNoise model to C code.

This script converts a trained PyTorch model (.pth) to C source files
that can be compiled for embedded deployment (ESP32, etc.).

The --quantize option converts float32 → int8 for size reduction.
The following layers are excluded from quantization: {UNQUANTIZED_LAYERS}
"""


def create_parser():
    """Create argument parser matching reference."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('weightfile', type=str, 
                       help='path to .pth checkpoint file')
    parser.add_argument('export_folder', type=str,
                       help='output folder for C files')
    parser.add_argument('--export-filename', type=str, 
                       default='rnnoise_data',
                       help='filename for C files (.c and .h will be added)')
    parser.add_argument('--struct-name', type=str, 
                       default='RNNoise',
                       help='name for C struct')
    parser.add_argument('--quantize', action='store_true',
                       help='apply int8 quantization (reduces model size)')
    
    return parser


def main():
    """Main export function - exact match to reference."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not HAS_WEXCHANGE:
        print("ERROR: Cannot export without weight-exchange library")
        return 1
    
    # Load checkpoint
    print(f"Loading weights from {args.weightfile}...")
    try:
        checkpoint = torch.load(args.weightfile, map_location='cpu')
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return 1
    
    # Ensure model_args and model_kwargs exist (from reference)
    if 'model_args' not in checkpoint:
        checkpoint['model_args'] = ()
    if 'model_kwargs' not in checkpoint:
        # Try to infer from state_dict
        print("WARNING: model_kwargs not found in checkpoint")
        checkpoint['model_kwargs'] = {}
    
    # Create model
    print("Creating model...")
    try:
        model = RNNoise(*checkpoint['model_args'], **checkpoint['model_kwargs'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    except Exception as e:
        print(f"ERROR: Failed to create model: {e}")
        return 1
    
    # Remove weight normalization if present (from reference)
    def _remove_weight_norm(m):
        try:
            torch.nn.utils.remove_weight_norm(m)
        except ValueError:
            return
    model.apply(_remove_weight_norm)
    
    # Create output directory
    os.makedirs(args.export_folder, exist_ok=True)
    
    # Initialize C writer (from reference)
    print(f"Dumping model to {args.export_folder}...")
    writer = wexchange.c_export.c_writer.CWriter(
        os.path.join(args.export_folder, args.export_filename),
        model_struct_name=args.struct_name,
        add_typedef=True
    )
    
    # Export each layer (exact logic from reference)
    quantize_model = args.quantize
    
    for name, module in model.named_modules():
        # Determine quantization settings
        if quantize_model:
            quantize = name not in UNQUANTIZED_LAYERS
            scale = None if quantize else 1/128
        else:
            quantize = False
            scale = 1/128
        
        # Export based on layer type
        if isinstance(module, nn.Linear):
            print(f"Dumping Linear layer {name}...")
            wexchange.torch.dump_torch_dense_weights(
                writer, module, 
                name.replace('.', '_'),
                quantize=quantize, 
                scale=scale
            )
        
        elif isinstance(module, nn.Conv1d):
            print(f"Dumping Conv1d layer {name}...")
            wexchange.torch.dump_torch_conv1d_weights(
                writer, module,
                name.replace('.', '_'),
                quantize=quantize,
                scale=scale
            )
        
        elif isinstance(module, nn.GRU):
            print(f"Dumping GRU layer {name}...")
            wexchange.torch.dump_torch_gru_weights(
                writer, module,
                name.replace('.', '_'),
                quantize=quantize,
                scale=scale,
                recurrent_scale=scale,
                input_sparse=True,      # Enable sparse format
                recurrent_sparse=True   # Enable sparse format
            )
        
        elif isinstance(module, nn.GRUCell):
            print(f"Dumping GRUCell layer {name}...")
            wexchange.torch.dump_torch_grucell_weights(
                writer, module,
                name.replace('.', '_'),
                quantize=quantize,
                scale=scale,
                recurrent_scale=scale
            )
        
        elif isinstance(module, nn.Embedding):
            print(f"Dumping Embedding layer {name}...")
            wexchange.torch.dump_torch_embedding_weights(
                writer, module,
                name.replace('.', '_'),
                quantize=quantize,
                scale=scale
            )
        
        else:
            # Skip container modules (RNNoise, Sequential, etc.)
            if name != '':  # Root module has empty name
                print(f"Ignoring layer {name}...")
    
    # Finalize
    writer.close()
    
    print(f"\n✅ Export complete!")
    print(f"   Output files:")
    print(f"   - {os.path.join(args.export_folder, args.export_filename)}.c")
    print(f"   - {os.path.join(args.export_folder, args.export_filename)}.h")
    
    if quantize_model:
        print(f"   Quantization: int8 (enabled)")
    else:
        print(f"   Quantization: float32 (disabled)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
