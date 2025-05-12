# api/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
import json

# Import your core functionality
from core.model_loader import load_model
from core.peft_manager import apply_peft
from utils.dataset import load_dataset, TextDataset
from core.trainer import train_model
from core.inference import generate_response

app = FastAPI(title="PEFT App API")

# Track active jobs
ACTIVE_JOBS = {}

# Models
class PeftConfig(BaseModel):
    peft_type: str = "lora"
    rank: int = 8
    alpha: int = 16
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.05

class TrainConfig(BaseModel):
    base_model: str
    peft_config: PeftConfig
    dataset_path: str
    output_dir: Optional[str] = None
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    quantize: bool = True

class GenerateConfig(BaseModel):
    model_path: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    details: Dict[str, Any] = {}

# Endpoints
@app.post("/train", response_model=JobStatus)
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    """Start a new PEFT training job"""
    job_id = str(uuid.uuid4())
    
    # Set output directory if not provided
    output_dir = config.output_dir or f"./models/{job_id}"
    
    # Initialize job status
    ACTIVE_JOBS[job_id] = {
        "status": "initializing",
        "progress": 0.0,
        "details": {
            "base_model": config.base_model,
            "peft_config": config.peft_config.dict(),
            "dataset_path": config.dataset_path,
            "output_dir": output_dir
        }
    }
    
    # Start training in background
    background_tasks.add_task(
        run_training_job,
        job_id=job_id,
        config=config,
        output_dir=output_dir
    )
    
    return JobStatus(
        job_id=job_id,
        status="initializing",
        progress=0.0,
        details=ACTIVE_JOBS[job_id]["details"]
    )

async def run_training_job(job_id: str, config: TrainConfig, output_dir: str):
    """Run the training job in the background"""
    try:
        # Update job status
        ACTIVE_JOBS[job_id]["status"] = "loading_base_model"
        ACTIVE_JOBS[job_id]["progress"] = 0.1
        
        # Load the base model
        model, tokenizer = load_model(
            config.base_model,
            load_in_8bit=config.quantize,
            load_in_4bit=False
        )
        
        # Update job status
        ACTIVE_JOBS[job_id]["status"] = "applying_peft"
        ACTIVE_JOBS[job_id]["progress"] = 0.2
        
        # Apply PEFT configuration
        peft_model = apply_peft(
            model,
            peft_type=config.peft_config.peft_type,
            rank=config.peft_config.rank,
            alpha=config.peft_config.alpha,
            target_modules=config.peft_config.target_modules,
            lora_dropout=config.peft_config.lora_dropout
        )
        
        # Update job status
        ACTIVE_JOBS[job_id]["status"] = "loading_dataset"
        ACTIVE_JOBS[job_id]["progress"] = 0.3
        
        # Load and prepare dataset
        data = load_dataset(config.dataset_path)
        dataset = TextDataset(data, tokenizer)
        
        # Update job status
        ACTIVE_JOBS[job_id]["status"] = "training"
        ACTIVE_JOBS[job_id]["progress"] = 0.4
        
        # Train the model
        train_model(
            peft_model,
            tokenizer,
            dataset,
            output_dir=output_dir,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate
        )
        
        # Update job status
        ACTIVE_JOBS[job_id]["status"] = "completed"
        ACTIVE_JOBS[job_id]["progress"] = 1.0
        ACTIVE_JOBS[job_id]["details"]["model_path"] = output_dir
        
    except Exception as e:
        # Update job status on error
        ACTIVE_JOBS[job_id]["status"] = "failed"
        ACTIVE_JOBS[job_id]["details"]["error"] = str(e)
        print(f"Training job {job_id} failed: {str(e)}")

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a training job"""
    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job_id,
        status=ACTIVE_JOBS[job_id]["status"],
        progress=ACTIVE_JOBS[job_id]["progress"],
        details=ACTIVE_JOBS[job_id]["details"]
    )

@app.post("/generate")
async def generate_text(config: GenerateConfig):
    """Generate text using a fine-tuned model"""
    try:
        # Load the model
        model, tokenizer = load_model(config.model_path)
        
        # Generate response
        response = generate_response(
            model,
            tokenizer,
            config.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature
        )
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile):
    """Upload a dataset file"""
    # Create temporary directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Save the uploaded file
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {"datasetPath": file_path}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


game_plan = """
Ok so 


couple moving parts right now:

* Cloud: lambdalabs, sagemaker
* Backend: PEFT, CoAR, Harmbench
* Connecting API: FastAPI (REST
* Frontend: React
"""

todo_list = """
TODO: figure out 

coAR:
I'm thinking I can do diffusion models as well or maybe increase support for multi modal models.

What does ilyas say in his paper:

Future work. We highlight three directions that, while outside the scope of this work, may be
interesting avenues for future work.

• Analyzing neural network representations. An interesting direction for future work could be
to use component attribution (and component models, more generally) to study empirically
documented phenomena in deep learning. 

There are a plethora of questions to ask here which,
although beyond the scope of this work, are natural applications of component attributions. For
example, extending our results from Section 5.1, can we use component attribution to better isolate “opposing signals” [RR23] 

for a given task, and to understand their role in shaping model
predictions? Can we use component attributions to study how model predictions change due
to adversarial perturbations [GSS15], or over the course of training [KKN+19]? 

Similarly, can
we develop improved methods for localizing memorized inputs to specific model components
[FZ20; MMS+23]? Given that component attributions are causally meaningful, can we use them
as a kernel with which to compare different models [KNL+19] or learning algorithms [SPI+23]?



• Attributing generative models. While we focus on vision models in this work, COAR is a general method that can estimate 
component attributions for any machine learning model. Future
work might thus explore possible model output functions (and their corresponding component
attributions) for generative models. 

For diffusion-based generative models, one might study the
denoising error for a fixed timestep, as in [GVS+23; ZPD+23]. 

For language models, a possible
point of start (following Park et al. [PGI+23]) would be to use the average correct-class margin (5) 
of a sequence of tokens as the model output function. Our preliminary experiments
in Appendix B show that COAR learns accurate component attributions for language models
such as GPT-2 [RWC+19] and Phi-2 [LBE+23]. 

In general, estimating and applying component
attributions for generative models is a promising avenue for future work.


• Beyond linear component attribution. The fact that component attributions’ predictiveness
decreases on out-of-distribution component subsets, i.e., when αtest ̸= αtrain, suggests that the
linear form of component attributions might not be expressive enough to fully capture the map
between model components and outputs. Given the generality of COAR, an interesting avenue
for future work would be to explore whether non-linear component models such as decision
trees or kernel methods predict component counterfactuals more accurately, and as a result,
improve model editing
"""
