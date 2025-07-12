#!/usr/bin/env python3
"""
Complete PELA Implementation with Feature Distillation and Fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm
import json
import os
from typing import Dict, Any, Optional, List
import gc

class PELADistillationLoss(nn.Module):
    """Feature distillation loss for PELA training."""
    
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits, labels=None):
        """Compute distillation loss."""
        # Temperature-scaled KL divergence
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        distillation_loss *= (self.temperature ** 2)
        
        # Combined with task loss if labels provided
        if labels is not None:
            task_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * task_loss + (1 - self.alpha) * distillation_loss
        else:
            total_loss = distillation_loss
            
        return total_loss

class OCRDistillationDataset(Dataset):
    """Simple dataset for OCR distillation training."""
    
    def __init__(self, images_and_texts: List[tuple], processor, max_length: int = 512):
        self.data = images_and_texts
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, text = self.data[idx]
        
        # Create input-output pair
        prompt = "Extract all text from this image:"
        target = text
        
        # Process inputs
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Process targets  
        targets = self.processor.tokenizer(
            target,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze() if 'pixel_values' in inputs else None,
            'labels': targets['input_ids'].squeeze()
        }

def create_synthetic_ocr_data(num_samples: int = 100):
    """Create synthetic OCR data for distillation training."""
    from PIL import Image, ImageDraw, ImageFont
    import random
    import string
    
    data = []
    
    for i in range(num_samples):
        # Create synthetic text
        text_length = random.randint(10, 50)
        text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=text_length))
        
        # Create simple text image
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        draw.text((10, 30), text, fill='black', font=font)
        
        data.append((img, text))
        
        if i % 20 == 0:
            print(f"Generated {i+1}/{num_samples} synthetic samples")
    
    return data

def distill_and_finetune(teacher_model_path: str, 
                        student_model_path: str,
                        num_epochs: int = 3,
                        batch_size: int = 2,
                        learning_rate: float = 1e-5,
                        output_path: str = "olmocr_distilled_1.5x.pt"):
    """
    Complete PELA distillation and fine-tuning process.
    """
    
    print("🎓 Starting PELA Distillation and Fine-tuning")
    print("=" * 60)
    
    # Load models
    print("📥 Loading teacher model...")
    if teacher_model_path.endswith('.pt'):
        teacher_model = torch.load(teacher_model_path, map_location='cpu', weights_only=False)
    else:
        teacher_model = Qwen2VLForConditionalGeneration.from_pretrained(teacher_model_path)
    teacher_model.eval()
    
    print("📥 Loading student model...")
    student_model = torch.load(student_model_path, map_location='cpu', weights_only=False)
    student_model.train()
    
    # Load processor
    processor = AutoProcessor.from_pretrained("allenai/olmOCR-7B-0225-preview")
    
    # Create training data
    print("🔧 Creating synthetic training data...")
    training_data = create_synthetic_ocr_data(num_samples=50)  # Small for demo
    dataset = OCRDistillationDataset(training_data, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    distillation_loss = PELADistillationLoss(alpha=0.7, temperature=3.0)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    print(f"🚀 Training on device: {device}")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels']
                )
                teacher_logits = teacher_outputs.logits
            
            # Forward pass through student
            student_outputs = student_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
            student_logits = student_outputs.logits
            
            # Compute distillation loss
            loss = distillation_loss(student_logits, teacher_logits, batch['labels'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            # Memory cleanup
            del teacher_outputs, student_outputs, teacher_logits, student_logits
            gc.collect()
            
            if num_batches % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = epoch_loss / num_batches
        print(f"📊 Average loss: {avg_loss:.4f}")
    
    # Save distilled model
    print(f"\n💾 Saving distilled model to {output_path}...")
    student_model = student_model.to('cpu')  # Move to CPU for saving
    torch.save(student_model, output_path)
    
    print("✅ Distillation and fine-tuning completed!")
    return output_path

def compare_all_models():
    """Compare original, compressed, and distilled models."""
    
    models_to_test = [
        ("Original", "models/olmOCR-7B-0225-preview"),
        ("Compressed (SVD only)", "olmocr_compressed_1.5x.pt"),
        ("Distilled (Complete PELA)", "olmocr_distilled_1.5x.pt")
    ]
    
    print("🔍 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    
    results = {}
    
    for model_name, model_path in models_to_test:
        print(f"\n📊 Testing {model_name}...")
        
        if not os.path.exists(model_path) and not model_path.startswith("models/"):
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        try:
            # Simple functionality test
            if model_path.endswith('.pt'):
                model = torch.load(model_path, map_location='cpu', weights_only=False)
            else:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            results[model_name] = {
                'total_params': total_params,
                'model_size_gb': total_params * 4 / (1024**3),  # Assume float32
                'loads_successfully': True
            }
            
            print(f"✅ Loads successfully")
            print(f"📊 Parameters: {total_params:,}")
            print(f"💾 Size: {results[model_name]['model_size_gb']:.2f} GB")
            
            del model
            gc.collect()
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[model_name] = {'loads_successfully': False, 'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("🚀 Complete PELA Implementation")
    print("Choose an option:")
    print("1. Test current compressed model on real OCR")
    print("2. Run distillation and fine-tuning")
    print("3. Compare all models")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        print("🧪 Testing compressed model...")
        os.system("python test_real_ocr.py")
    
    elif choice == "2":
        print("🎓 Running complete PELA distillation...")
        distilled_path = distill_and_finetune(
            teacher_model_path="models/olmOCR-7B-0225-preview",
            student_model_path="olmocr_compressed_1.5x.pt",
            num_epochs=2,  # Quick demo
            output_path="olmocr_distilled_1.5x.pt"
        )
        print(f"✅ Distilled model saved: {distilled_path}")
    
    elif choice == "3":
        print("📊 Comparing all models...")
        results = compare_all_models()
        print("\n📋 SUMMARY:")
        for name, stats in results.items():
            if stats.get('loads_successfully'):
                print(f"{name}: {stats['model_size_gb']:.2f} GB, {stats['total_params']:,} params")
            else:
                print(f"{name}: FAILED")
    
    else:
        print("Invalid choice. Run script again.") 