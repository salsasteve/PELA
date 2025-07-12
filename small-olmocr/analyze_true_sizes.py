#!/usr/bin/env python3
"""
Accurate file size analysis for PELA compression results
"""

import os
import glob
from pathlib import Path

def get_directory_size(path):
    """Get total size of all files in directory"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total

def get_file_size(path):
    """Get size of a single file"""
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0

def format_bytes(bytes_val):
    """Format bytes as human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def find_original_model():
    """Try to find the original model in various locations"""
    possible_locations = [
        "../models/allenai--OLMoOCR-7B-0225-preview",
        "./models/allenai--OLMoOCR-7B-0225-preview",
        "models/allenai--OLMoOCR-7B-0225-preview",
        os.path.expanduser("~/.cache/huggingface/hub/models--allenai--OLMoOCR-7B-0225-preview"),
        os.path.expanduser("~/.cache/huggingface/transformers/models--allenai--OLMoOCR-7B-0225-preview"),
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            return location
    return None

def analyze_models():
    print("🔍 ACCURATE MODEL SIZE ANALYSIS")
    print("=" * 60)
    
    # Find original model
    original_path = find_original_model()
    if original_path:
        original_size = get_directory_size(original_path)
        print(f"📁 Original model found at: {original_path}")
        print(f"📁 Original model size: {format_bytes(original_size)}")
        
        # List major files
        print("   📋 Major files:")
        for file_path in glob.glob(os.path.join(original_path, "*")):
            if os.path.isfile(file_path):
                size = get_file_size(file_path)
                if size > 1024 * 1024:  # Only show files > 1MB
                    print(f"      {os.path.basename(file_path)}: {format_bytes(size)}")
    else:
        print("❌ Original model not found in any expected location")
        print("   Searched:")
        possible_locations = [
            "../models/allenai--OLMoOCR-7B-0225-preview",
            "./models/allenai--OLMoOCR-7B-0225-preview", 
            "models/allenai--OLMoOCR-7B-0225-preview",
            os.path.expanduser("~/.cache/huggingface/hub/models--allenai--OLMoOCR-7B-0225-preview"),
        ]
        for loc in possible_locations:
            print(f"      {loc}")
        original_size = 0
    
    print()
    
    # Check local models directory
    local_models_dir = "./models"
    if os.path.exists(local_models_dir):
        print(f"📁 Local models directory: {local_models_dir}")
        for file_path in glob.glob(os.path.join(local_models_dir, "*")):
            if os.path.isfile(file_path):
                size = get_file_size(file_path)
                filename = os.path.basename(file_path)
                print(f"   {filename}: {format_bytes(size)}")
                
                if original_size > 0:
                    ratio = size / original_size
                    print(f"      📊 vs Original: {ratio:.2f}x ({'larger' if ratio > 1 else 'smaller'})")
    
    print("\n🎯 COMPRESSION ANALYSIS")
    print("=" * 60)
    
    if original_size > 0:
        print(f"Original model: {format_bytes(original_size)}")
        
        compressed_files = glob.glob("./models/olmocr_compressed_*.pt")
        for comp_file in compressed_files:
            comp_size = get_file_size(comp_file)
            ratio = comp_size / original_size
            savings = original_size - comp_size
            filename = os.path.basename(comp_file)
            
            print(f"\n{filename}:")
            print(f"  Size: {format_bytes(comp_size)}")
            print(f"  Ratio: {ratio:.2f}x")
            
            if ratio < 1:
                print(f"  ✅ Savings: {format_bytes(savings)} ({(1-ratio)*100:.1f}% reduction)")
            else:
                print(f"  ❌ Increase: {format_bytes(-savings)} ({(ratio-1)*100:.1f}% larger)")
    else:
        print("⚠️  Cannot calculate compression ratios without original model size")
        print("📋 Compressed model sizes:")
        compressed_files = glob.glob("./models/olmocr_compressed_*.pt")
        for comp_file in compressed_files:
            comp_size = get_file_size(comp_file)
            filename = os.path.basename(comp_file)
            print(f"   {filename}: {format_bytes(comp_size)}")

if __name__ == "__main__":
    analyze_models() 