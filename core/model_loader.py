# core/model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_id, device_map="auto", load_in_8bit=False, load_in_4bit=False):
    """Load base model with optional quantization"""
    kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    
    # Apply quantization if requested
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["bnb_4bit_compute_dtype"] = torch.float16
        kwargs["bnb_4bit_quant_type"] = "nf4"
    else:
        kwargs["torch_dtype"] = torch.float16
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer