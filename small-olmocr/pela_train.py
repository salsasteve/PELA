"""
PELA Training Script for OLMoOCR

This script extends the standard OLMoOCR training to include PELA 
(Parameter-Efficient Learning with Low-Rank Approximation) with 
feature distillation and weight perturbation regularization.
"""

import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from dataclasses import dataclass
import yaml

from pela_olmocr import compress_olmocr_with_pela, ModulePELA

logger = logging.getLogger(__name__)


@dataclass
class PELAConfig:
    """Configuration for PELA training."""
    enabled: bool = True
    compress_ratio: float = 3.0
    min_rank: int = 8
    max_rank: int = 256
    
    # Feature distillation parameters
    distillation_alpha: float = 0.5
    distillation_temperature: float = 4.0
    distillation_weight: float = 1.0
    
    # Weight perturbation regularization
    weight_perturbation_weight: float = 0.1
    perturbation_std: float = 0.01
    
    # Training parameters
    freeze_original_weights: bool = True
    target_modules: Optional[list] = None
    exclude_modules: Optional[list] = None


class PELAFeatureDistillationLoss(nn.Module):
    """
    Feature distillation loss for PELA training.
    
    Computes distillation loss between original model and PELA-compressed model
    at multiple levels: hidden states, attention maps, and final outputs.
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 temperature: float = 4.0,
                 feature_layers: Optional[list] = None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feature_layers = feature_layers or [-1]  # Last layer by default
        
    def forward(self, 
                student_outputs,
                teacher_outputs,
                student_features: Optional[Dict] = None,
                teacher_features: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute distillation loss between student (PELA) and teacher (original) models.
        
        Args:
            student_outputs: Outputs from PELA model
            teacher_outputs: Outputs from original model
            student_features: Intermediate features from PELA model
            teacher_features: Intermediate features from original model
            
        Returns:
            Combined distillation loss
        """
        total_loss = 0.0
        loss_count = 0
        
        # Output distillation loss
        if hasattr(student_outputs, 'logits') and hasattr(teacher_outputs, 'logits'):
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
            
            # Temperature-scaled softmax distillation
            student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
            total_loss += kl_loss * (self.temperature ** 2)
            loss_count += 1
        
        # Feature distillation loss
        if student_features and teacher_features:
            for layer_idx in self.feature_layers:
                if (layer_idx in student_features and layer_idx in teacher_features):
                    student_feat = student_features[layer_idx]
                    teacher_feat = teacher_features[layer_idx]
                    
                    # MSE loss between features
                    feat_loss = F.mse_loss(student_feat, teacher_feat)
                    total_loss += feat_loss
                    loss_count += 1
        
        return total_loss / max(loss_count, 1)


class PELAWeightPerturbationLoss(nn.Module):
    """
    Weight perturbation regularization for PELA training.
    
    Adds small perturbations to weights and encourages stability
    to improve generalization of the compressed model.
    """
    
    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std
        
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute weight perturbation regularization loss.
        
        Args:
            model: PELA model
            
        Returns:
            Regularization loss
        """
        total_loss = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and 'linear_' in name:  # Target PELA layers
                # Add small perturbation
                noise = torch.randn_like(param) * self.std
                perturbed_param = param + noise
                
                # L2 regularization on perturbation
                reg_loss = torch.norm(perturbed_param - param, p=2)
                total_loss += reg_loss
                param_count += 1
        
        return total_loss / max(param_count, 1)


class PELATrainer(Trainer):
    """
    Extended Trainer class for PELA training with feature distillation
    and weight perturbation regularization.
    """
    
    def __init__(self, 
                 pela_config: PELAConfig,
                 teacher_model: Optional[nn.Module] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.pela_config = pela_config
        self.teacher_model = teacher_model
        
        # Initialize loss functions
        self.distillation_loss = PELAFeatureDistillationLoss(
            alpha=pela_config.distillation_alpha,
            temperature=pela_config.distillation_temperature
        )
        
        self.perturbation_loss = PELAWeightPerturbationLoss(
            std=pela_config.perturbation_std
        )
        
        # Move teacher model to same device if provided
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute PELA training loss including feature distillation and regularization.
        """
        # Forward pass through student (PELA) model
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss if hasattr(student_outputs, 'loss') else 0.0
        
        total_loss = student_loss
        
        # Feature distillation with teacher model
        if self.teacher_model is not None and self.pela_config.distillation_weight > 0:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            
            distill_loss = self.distillation_loss(student_outputs, teacher_outputs)
            total_loss += self.pela_config.distillation_weight * distill_loss
        
        # Weight perturbation regularization
        if self.pela_config.weight_perturbation_weight > 0:
            perturb_loss = self.perturbation_loss(model)
            total_loss += self.pela_config.weight_perturbation_weight * perturb_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss


class PELATrainingCallback(TrainerCallback):
    """Callback for monitoring PELA training progress."""
    
    def __init__(self, pela_config: PELAConfig, log_every: int = 100):
        self.pela_config = pela_config
        self.log_every = log_every
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log PELA-specific metrics."""
        if state.is_local_process_zero and state.global_step % self.log_every == 0:
            model = kwargs.get('model')
            if model:
                # Count PELA parameters
                pela_params = sum(
                    p.numel() for name, p in model.named_parameters() 
                    if p.requires_grad and 'linear_' in name
                )
                
                logger.info(f"Step {state.global_step}: PELA trainable parameters: {pela_params:,}")


def setup_pela_training(model, config: PELAConfig, logger=None):
    """
    Setup model for PELA training.
    
    Args:
        model: Original OLMoOCR model
        config: PELA configuration
        logger: Logger instance
        
    Returns:
        Tuple of (teacher_model, student_model)
    """
    if logger:
        logger.info("Setting up PELA training...")
    
    # Keep original model as teacher
    teacher_model = model
    
    # Create PELA-compressed student model
    student_model = compress_olmocr_with_pela(
        model,
        compress_ratio=config.compress_ratio,
        target_modules=config.target_modules,
        exclude_modules=config.exclude_modules,
        min_rank=config.min_rank,
        max_rank=config.max_rank
    )
    
    # Freeze original weights if specified
    if config.freeze_original_weights:
        frozen_count = 0
        trainable_count = 0
        
        for name, param in student_model.named_parameters():
            # Only keep PELA layers trainable
            if 'linear_l' not in name and 'linear_r' not in name:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True  # Ensure PELA layers are trainable
                trainable_count += param.numel()
        
        if logger:
            total_params = frozen_count + trainable_count
            logger.info(f"Frozen original weights. Trainable: {trainable_count:,} / {total_params:,} "
                       f"({100 * trainable_count / total_params:.1f}%)")
            logger.info(f"  🧊 Frozen parameters: {frozen_count:,} ({100 * frozen_count / total_params:.1f}%)")
            logger.info(f"  🔥 PELA trainable: {trainable_count:,} ({100 * trainable_count / total_params:.1f}%)")
    
    return teacher_model, student_model


def main_pela_train(model_name_or_path: str,
                   pela_config_path: str,
                   training_args: TrainingArguments,
                   train_dataset,
                   eval_dataset=None,
                   processor=None):
    """
    Main PELA training function.
    
    Args:
        model_name_or_path: Path to the pre-trained model
        pela_config_path: Path to PELA configuration file
        training_args: HuggingFace training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        processor: Model processor
    """
    # Load PELA configuration
    with open(pela_config_path, 'r') as f:
        pela_config_dict = yaml.safe_load(f)
    pela_config = PELAConfig(**pela_config_dict.get('pela', {}))
    
    logger.info(f"Loading model from {model_name_or_path}")
    
    # Load original model
    if "qwen" in model_name_or_path.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16
        )
    else:
        from olmocr.train.molmo.config_molmo import MolmoConfig
        from olmocr.train.molmo.modeling_molmo import MolmoForCausalLM
        
        model_config = MolmoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = MolmoForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            config=model_config, 
            trust_remote_code=True
        )
    
    # Setup PELA training
    teacher_model, student_model = setup_pela_training(model, pela_config, logger)
    
    # Initialize PELA trainer
    trainer = PELATrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer if processor else None,
        pela_config=pela_config,
        teacher_model=teacher_model,
        callbacks=[PELATrainingCallback(pela_config)]
    )
    
    logger.info("Starting PELA training...")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Save PELA statistics
    if hasattr(trainer.model, 'pela_stats'):
        stats_path = os.path.join(training_args.output_dir, 'pela_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(trainer.model.pela_stats, f, indent=2)
        logger.info(f"PELA statistics saved to {stats_path}")
    
    return trainer


if __name__ == "__main__":
    # Example usage
    import argparse
    from transformers import AutoProcessor
    
    parser = argparse.ArgumentParser(description="PELA Training for OLMoOCR")
    parser.add_argument("--model_name_or_path", required=True, help="Model name or path")
    parser.add_argument("--pela_config", required=True, help="PELA configuration file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--train_data", required=True, help="Training data path")
    parser.add_argument("--eval_data", help="Evaluation data path")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        bf16=True,
        report_to="none"  # Disable wandb for example
    )
    
    # For this example, create dummy datasets
    # In practice, you would load your actual OCR training data
    print("Note: This example uses dummy datasets. Replace with actual OCR data loading.")
    
    # Run PELA training
    # trainer = main_pela_train(
    #     model_name_or_path=args.model_name_or_path,
    #     pela_config_path=args.pela_config,
    #     training_args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     processor=processor
    # ) 