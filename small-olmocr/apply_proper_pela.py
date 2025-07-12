#!/usr/bin/env python3
"""
Apply proper aggressive PELA compression to real OLMoOCR model
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from proper_pela import AggressivePELACompressor
import time
import os

def load_olmocr_model():
    """Load the OLMoOCR model"""
    print("📥 Loading OLMoOCR model...")
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/OLMoOCR-7B-0225-preview",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print(f"✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def analyze_original_model(model):
    """Analyze the original model before compression"""
    print("\n🔍 ANALYZING ORIGINAL MODEL")
    print("=" * 50)
    
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    linear_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_params += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                linear_params += module.bias.numel()
            linear_layers += 1
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Linear parameters: {linear_params:,} ({linear_params/total_params*100:.1f}%)")
    print(f"📊 Linear layers: {linear_layers}")
    print(f"📊 Model size (fp32): {total_params * 4 / (1024**3):.1f} GB")
    print(f"📊 Model size (fp16): {total_params * 2 / (1024**3):.1f} GB")
    
    return total_params, linear_params, linear_layers

def apply_pela_compression(model, target_compression=1.5):
    """Apply PELA compression to the model"""
    print(f"\n🚀 APPLYING PELA COMPRESSION (target: {target_compression}x)")
    print("=" * 70)
    
    # Create compressor
    compressor = AggressivePELACompressor(target_compression=target_compression)
    
    # Compress the model
    start_time = time.time()
    compressed_model = compressor.compress_model(model)
    compression_time = time.time() - start_time
    
    print(f"\n⏱️  Compression completed in {compression_time:.1f} seconds")
    
    return compressed_model, compressor.compression_stats

def analyze_compressed_model(model, original_params):
    """Analyze the compressed model"""
    print(f"\n🔍 ANALYZING COMPRESSED MODEL")
    print("=" * 50)
    
    compressed_params = sum(p.numel() for p in model.parameters())
    compression_ratio = original_params / compressed_params
    
    print(f"📊 Compressed parameters: {compressed_params:,}")
    print(f"📊 Compression ratio: {compression_ratio:.1f}x")
    print(f"📊 Parameters saved: {original_params - compressed_params:,}")
    print(f"📊 Size reduction: {(1 - compressed_params/original_params)*100:.1f}%")
    print(f"📊 Compressed size (fp16): {compressed_params * 2 / (1024**3):.1f} GB")
    
    return compressed_params, compression_ratio

def save_compressed_model(model, target_compression, actual_compression):
    """Save the compressed model efficiently"""
    print(f"\n💾 SAVING COMPRESSED MODEL")
    print("=" * 50)
    
    # Convert to fp16 and save state_dict only
    model = model.half()
    output_path = f"./models/olmocr_proper_pela_{target_compression}x.pt"
    
    print(f"🔄 Converting to fp16 and saving state_dict...")
    torch.save(model.state_dict(), output_path)
    
    # Check file size
    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"✅ Saved to: {output_path}")
    print(f"📁 File size: {file_size:.1f} GB")
    
    return output_path, file_size

def test_model_functionality(model):
    """Quick test to ensure the compressed model still works"""
    print(f"\n🧪 TESTING MODEL FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Test forward pass with dummy input
        model.eval()
        with torch.no_grad():
            # Create dummy input (simplified)
            batch_size = 1
            seq_len = 10
            dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            dummy_attention_mask = torch.ones(batch_size, seq_len)
            
            # Forward pass
            outputs = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            )
            
            print(f"✅ Forward pass successful")
            print(f"📊 Output shape: {outputs.logits.shape}")
            print(f"📊 Output dtype: {outputs.logits.dtype}")
            
            return True
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def main():
    """Main compression pipeline"""
    print("🎯 PROPER PELA COMPRESSION PIPELINE")
    print("=" * 60)
    
    # Load model
    model = load_olmocr_model()
    if model is None:
        return
    
    # Analyze original
    original_params, linear_params, linear_layers = analyze_original_model(model)
    
    # Test different compression ratios
    targets = [1.5, 2.0]
    
    for target in targets:
        print(f"\n{'='*80}")
        print(f"🎯 TESTING {target}x COMPRESSION")
        print(f"{'='*80}")
        
        # Make a copy for compression
        model_copy = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/OLMoOCR-7B-0225-preview",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Apply compression
        compressed_model, stats = apply_pela_compression(model_copy, target)
        
        # Analyze results
        compressed_params, actual_ratio = analyze_compressed_model(compressed_model, original_params)
        
        # Test functionality
        works = test_model_functionality(compressed_model)
        
        if works:
            # Save the model
            output_path, file_size = save_compressed_model(compressed_model, target, actual_ratio)
            
            print(f"\n🏆 FINAL RESULTS FOR {target}x TARGET:")
            print(f"   ✅ Compression achieved: {actual_ratio:.1f}x")
            print(f"   ✅ Model file: {file_size:.1f} GB")
            print(f"   ✅ Functionality: Working")
            
            # Compare to your old results
            if target == 1.5:
                old_size = 15.1  # From your previous attempt
                improvement = (old_size - file_size) / old_size * 100
                print(f"   🚀 vs Old PELA: {improvement:.1f}% smaller!")
        else:
            print(f"   ❌ Model broken - compression too aggressive")

if __name__ == "__main__":
    main() 