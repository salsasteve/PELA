#!/usr/bin/env python3
"""
Download OLMoOCR model using huggingface-hub (more reliable than git clone)
"""

import os
from huggingface_hub import snapshot_download
from tqdm import tqdm
import shutil

def download_olmocr_model():
    """Download the OLMoOCR model to local models directory."""
    
    model_name = "allenai/olmOCR-7B-0225-preview"
    local_dir = "./models/olmOCR-7B-0225-preview"
    
    print(f"🚀 Downloading {model_name} to {local_dir}")
    print("📦 This may take a while (model is ~7GB)...")
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Remove existing directory if it exists and is incomplete
    if os.path.exists(local_dir):
        print(f"🗑️  Removing existing incomplete download at {local_dir}")
        shutil.rmtree(local_dir)
    
    try:
        # Download with progress bar
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            tqdm_class=tqdm
        )
        
        print(f"✅ Successfully downloaded {model_name}")
        print(f"📁 Model available at: {local_dir}")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(local_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                size = os.path.getsize(os.path.join(root, file))
                size_mb = size / (1024 * 1024)
                print(f"{subindent}{file} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Try running: pip install huggingface-hub")
        return False

if __name__ == "__main__":
    success = download_olmocr_model()
    if success:
        print("\n🎉 Ready to run PELA compression test!")
        print("💡 Run: python test_pela_real.py")
    else:
        print("\n⚠️  Download failed. Please check your internet connection and try again.") 