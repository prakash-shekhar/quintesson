# core/trainer.py
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os

def train_model(
    model, 
    tokenizer, 
    dataset,
    output_dir="./peft_model",
    epochs=3,
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10
):
    """Train the PEFT model using the HF Trainer"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="paged_adamw_8bit"
    )
    
    # Data collator for batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete! Model saved to {output_dir}")
    return model