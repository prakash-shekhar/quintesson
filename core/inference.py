# core/inference.py
import torch

def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    """Generate a response using the fine-tuned model"""
    # Format prompt if needed
    if not prompt.startswith(("### Instruction:", "### User:")):
        # Apply default formatting - adjust based on your model's training format
        prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate with specified parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part
    response = generated_text[len(prompt):].strip()
    
    return response