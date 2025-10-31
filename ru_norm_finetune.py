base_model = "google/gemma-3-270m-it" # @param ["google/gemma-3-270m-it","google/gemma-3-1b-it","google/gemma-3-4b-it","google/gemma-3-12b-it","google/gemma-3-27b-it"] {"allow-input":true}
checkpoint_dir = "./ru_norm_gemma"
learning_rate = 1e-6

from datasets import load_dataset

# Load dataset from the Hub
dataset = load_dataset("kenenbek/gemma-russian-normalization-dataset")
print(dataset["train"][0]["messages"])
print(dataset["validation"][0]["messages"])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="cuda",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"Device: {model.device}")
print(f"DType: {model.dtype}")

from trl import SFTConfig

torch_dtype = model.dtype

args = SFTConfig(
    output_dir=checkpoint_dir,              # directory to save and repository id
    max_length=256,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=10,                     # number of training epochs
    per_device_train_batch_size=16,          # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    report_to="wandb",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # Template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    },

    logging_steps=100,                   
    save_strategy="steps",         # save every N steps
    save_steps=500,                # save checkpoint every 500 steps
    save_total_limit=5,            # keep only 5 most recent/best checkpoints
    eval_strategy="steps",         # run evaluation every N steps
    eval_steps=500,                # evaluate every 500 steps
    load_best_model_at_end=True,   # reload the best checkpoint at end
    metric_for_best_model="eval_loss",
    greater_is_better=False,       # smaller eval_loss = better model
)

from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    processing_class=tokenizer,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()


