# core/peft_manager.py
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

def apply_peft(
    model, 
    peft_type="lora", 
    rank=8, 
    alpha=16,
    target_modules=None,
    lora_dropout=0.05
):
    """Apply PEFT adapter to model"""
    # Prepare model for training if using quantization
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    
    # Auto-detect target modules if not specified
    if target_modules is None:
        target_modules = auto_detect_target_modules(model)
    
    # Configure PEFT (currently supports LoRA)
    if peft_type.lower() == "lora":
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        raise ValueError(f"PEFT type {peft_type} not supported")
    
    # Apply PEFT
    peft_model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    print_trainable_parameters(peft_model)
    
    return peft_model

def auto_detect_target_modules(model):
    """Automatically detect which modules to target based on model architecture"""
    # Common target modules for different model families
    target_modules = []
    
    # Check for Llama-like models
    if any(name.endswith('q_proj') for name, _ in model.named_modules()):
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    # Check for Mistral-like models
    elif any(name.endswith('W_q') for name, _ in model.named_modules()):
        target_modules = ["W_q", "W_v", "W_k", "W_o"]
    # Check for Phi-like models
    elif any(name.endswith('Wqkv') for name, _ in model.named_modules()):
        target_modules = ["Wqkv", "out_proj"]
    # Fallback for other models
    else:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.shape[0] > 256:
                parts = name.split('.')
                if parts:
                    target_modules.append(parts[-1])
    
    # Deduplicate
    target_modules = list(set(target_modules))
    print(f"Auto-detected target modules: {target_modules}")
    
    return target_modules

def print_trainable_parameters(model):
    """Print number of trainable parameters"""
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/all_params:.2%})")


