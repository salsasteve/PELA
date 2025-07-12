#!/usr/bin/env python3
"""
Debug why compressed model is larger than original
"""

import torch
import os
from transformers import Qwen2VLForConditionalGeneration

def analyze_model_size(model_path, model_name):
    """Analyze model size and structure."""
    
    print(f"\n🔍 Analyzing {model_name}")
    print("=" * 50)
    
    # File size
    if os.path.exists(model_path):
        file_size_gb = os.path.getsize(model_path) / (1024**3)
        print(f"📁 File size: {file_size_gb:.2f} GB")
    else:
        print(f"📁 Directory-based model")
    
    # Load model
    try:
        if model_path.endswith('.pt'):
            model = torch.load(model_path, map_location='cpu', weights_only=False)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
        
        print(f"✅ Loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None
    
    # Analyze parameters
    total_params = 0
    pela_layers = 0
    regular_layers = 0
    
    layer_types = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'numel'):
            params = module.weight.numel()
            total_params += params
            
            # Check if it's a PELA layer
            if hasattr(module, 'linear_l') and hasattr(module, 'linear_r'):
                pela_layers += 1
                layer_type = "PELA"
                # Also count the sub-layers
                if hasattr(module.linear_l, 'weight'):
                    params += module.linear_l.weight.numel()
                if hasattr(module.linear_r, 'weight'):
                    params += module.linear_r.weight.numel()
            else:
                regular_layers += 1
                layer_type = module.__class__.__name__
            
            if layer_type not in layer_types:
                layer_types[layer_type] = {'count': 0, 'params': 0}
            layer_types[layer_type]['count'] += 1
            layer_types[layer_type]['params'] += params
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Model memory: {total_params * 4 / (1024**3):.2f} GB (float32)")
    print(f"🔧 PELA layers: {pela_layers}")
    print(f"🔧 Regular layers: {regular_layers}")
    
    print(f"\n📋 Layer type breakdown:")
    for layer_type, stats in sorted(layer_types.items()):
        print(f"  {layer_type}: {stats['count']} layers, {stats['params']:,} params")
    
    # Check for duplicates
    print(f"\n🔍 Checking for duplicate layers...")
    module_names = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            module_names.append(name)
    
    # Look for patterns that suggest duplication
    suspicious_patterns = []
    for name in module_names:
        if any(pattern in name for pattern in ['linear_l', 'linear_r']):
            base_name = name.replace('.linear_l', '').replace('.linear_r', '')
            if base_name in module_names:
                suspicious_patterns.append((base_name, name))
    
    if suspicious_patterns:
        print(f"⚠️  Found {len(suspicious_patterns)} potential duplicate patterns:")
        for base, pela in suspicious_patterns[:5]:  # Show first 5
            print(f"    {base} -> {pela}")
    else:
        print(f"✅ No obvious duplicates found")
    
    return {
        'total_params': total_params,
        'pela_layers': pela_layers,
        'regular_layers': regular_layers,
        'layer_types': layer_types,
        'file_size_gb': file_size_gb if os.path.exists(model_path) else None
    }

def compare_models():
    """Compare all available models."""
    
    models_to_check = [
        ("Original", "models/olmOCR-7B-0225-preview"),
        ("Compressed 1.5x", "olmocr_compressed_1.5x.pt"),
        ("Compressed 9.5x", "olmocr_compressed_9.5x.pt")
    ]
    
    results = {}
    
    for name, path in models_to_check:
        if os.path.exists(path) or path.startswith("models/"):
            results[name] = analyze_model_size(path, name)
        else:
            print(f"\n⚠️  {name} not found at {path}")
    
    # Compare results
    print(f"\n🔍 COMPARISON SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"{'Model':<20} {'File Size':<12} {'Parameters':<15} {'PELA':<8} {'Regular':<8}")
        print("-" * 70)
        
        for name, stats in results.items():
            if stats:
                file_size = f"{stats['file_size_gb']:.1f} GB" if stats['file_size_gb'] else "N/A"
                params = f"{stats['total_params']:,}"
                pela = str(stats['pela_layers'])
                regular = str(stats['regular_layers'])
                
                print(f"{name:<20} {file_size:<12} {params:<15} {pela:<8} {regular:<8}")
    
    return results

def diagnose_compression_bug():
    """Diagnose why compression isn't working."""
    
    print(f"\n🐛 DIAGNOSING COMPRESSION BUG")
    print("=" * 50)
    
    # Check if compressed models exist
    compressed_files = ["olmocr_compressed_1.5x.pt", "olmocr_compressed_9.5x.pt"]
    
    for filename in compressed_files:
        if os.path.exists(filename):
            print(f"\n📋 Analyzing {filename}...")
            
            model = torch.load(filename, map_location='cpu', weights_only=False)
            
            # Count different layer types
            linear_layers = []
            pela_layers = []
            
            for name, module in model.named_modules():
                if hasattr(module, 'linear_l') and hasattr(module, 'linear_r'):
                    pela_layers.append(name)
                elif hasattr(module, 'weight') and 'linear' in module.__class__.__name__.lower():
                    linear_layers.append(name)
            
            print(f"  📊 PELA layers: {len(pela_layers)}")
            print(f"  📊 Regular linear layers: {len(linear_layers)}")
            
            # Check if any layer names overlap
            pela_base_names = set()
            for name in pela_layers:
                pela_base_names.add(name)
            
            overlapping = []
            for name in linear_layers:
                if name in pela_base_names:
                    overlapping.append(name)
            
            if overlapping:
                print(f"  🚨 BUG FOUND: {len(overlapping)} layers exist in BOTH forms!")
                print(f"     Examples: {overlapping[:3]}")
            else:
                print(f"  ✅ No layer duplication detected")
            
            del model

if __name__ == "__main__":
    print("🔍 Model Size Debugging Tool")
    
    # Run full comparison
    results = compare_models()
    
    # Diagnose compression issues
    diagnose_compression_bug()
    
    print(f"\n💡 DIAGNOSIS:")
    if results:
        # Check if compressed models are actually larger
        original_size = None
        compressed_sizes = []
        
        for name, stats in results.items():
            if stats and stats['file_size_gb']:
                if 'Original' in name:
                    original_size = stats['file_size_gb']
                elif 'Compressed' in name:
                    compressed_sizes.append(stats['file_size_gb'])
        
        if original_size and compressed_sizes:
            for comp_size in compressed_sizes:
                if comp_size > original_size:
                    print(f"🚨 CONFIRMED BUG: Compressed model ({comp_size:.1f}GB) > Original ({original_size:.1f}GB)")
                    print(f"💡 Likely cause: Original layers not being removed during compression")
                    print(f"🔧 Fix: Ensure old layers are properly deleted before adding PELA layers")
                else:
                    print(f"✅ Compression working: {comp_size:.1f}GB < {original_size:.1f}GB") 