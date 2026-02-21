# Import foundation tools
import os, shutil
import sys
import subprocess
import json
import logging
import gc
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, HfApi, create_repo
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Optimize small RAM memory to prevent OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
#Tell Hugging Face lib to wait
os.environ["HF_HUB_READ_TIMEOUT"] = "3600"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1000"

# Import PyTorch and Unsloth/OpenSloth (Because I have no money for Unsloth Premium)
from opensloth.patching.ddp_patch import ddp_patch
ddp_patch()

from unsloth import FastLanguageModel

import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.tensorboard import SummaryWriter

if not hasattr(torch, "int1"):
    torch.int1 = torch.int8 # Shim to prevent the AttributeError

# Import other tools
import pandas as pd
from typing import Any
from datasets import Dataset, load_dataset, Features, Value
from tqdm.auto import tqdm

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel
from transformers import TrainingArguments, PreTrainedTokenizerFast, TrainerCallback, DataCollatorForLanguageModeling
from accelerate import PartialState

def authenticate_hf(token_name):
    """
    Authenticates with Hugging Face using a Kaggle secret.
    Returns the token if successful, otherwise raises an error.
    """
    try:
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret(token_name)
        login(token=hf_token)
        return hf_token
    
    except Exception as e:        
        raise RuntimeError(f"Could not authenticate: {e}")

def prepare_hf_dataset(df_path, tokenizer, skip_count, is_tokenized=False, streaming=False):
    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are mimicking the chat style of {3}.<|eot_id|>"
        "<|start_header_id|>User - {0}<|end_header_id|>\n\n"
        "Datetime - [{1}] User's Message - #####USER#####{2}<|eot_id|>"
        "<|start_header_id|>Assistant - {3}<|end_header_id|>\n\n"
        "Datetime - [{4}] Assistant's Message -#####RESPONSE#####{5}"
    )
    
    EOS_TOKEN = tokenizer.eos_token
    
    # Load and Clean Data
    df_json = pd.read_json(df_path)
    df = df_json[["date", "from", "text"]].copy()
    del df_json
    gc.collect()

    df['text'] = df['text'].apply(lambda x: "".join([t['text'] if isinstance(t, dict) else t for t in x]) if isinstance(x, list) else str(x))
    df = df[df["text"].str.strip() != ""]
    df = df[df["from"].notna()]
    df = df.astype(str)

    if skip_count > 0:
        df = df.iloc[skip_count:].reset_index(drop=True)

    # Pairing 2 consecutive messages - We zip current (i) with next (i+1)
    u_df = df.iloc[:-1].reset_index(drop=True) 
    a_df = df.iloc[1:].reset_index(drop=True)
    
    df_pairs = pd.DataFrame({
        "u_from": u_df["from"], "u_date": u_df["date"], "u_text": u_df["text"],
        "a_from": a_df["from"], "a_date": a_df["date"], "a_text": a_df["text"]
    })
    
    dataset = Dataset.from_pandas(df_pairs)
    del df, u_df, a_df, df_pairs # Clean up memory
    gc.collect()
    
    # The Format Function
    def format_batch_fn(examples):
        formatted = [
            prompt_template.format(u_f, u_d, u_t, a_f, a_d, a_t) + EOS_TOKEN
            for u_f, u_d, u_t, a_f, a_d, a_t in zip(
                examples["u_from"], examples["u_date"], examples["u_text"],
                examples["a_from"], examples["a_date"], examples["a_text"]
            )
        ]
        return {"text": formatted}

    map_kwargs = {"batched": True, "num_proc": 2}
    
    # Apply streaming if requested
    if streaming:
        dataset = dataset.to_iterable_dataset()

    if is_tokenized:
        dataset = dataset.map(
            format_batch_fn, 
            **map_kwargs,
            desc="Formatting evaluation prompts"
        )
        
        # Define the tokenization logic
        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=512, 
                padding="max_length"
            )
            
            all_labels = []
            response_prefix = [68431, 68883, 26289, 68431] # Your Assistant Message marker
            
            for input_ids in tokenized["input_ids"]:
                labels = list(input_ids)
                prefix_idx = 0
                # Scan backwards to find the prompt/response split
                for i in range(len(labels) - len(response_prefix), -1, -1):
                    if labels[i : i + len(response_prefix)] == response_prefix:
                        prefix_idx = i + len(response_prefix)
                        break
                
                # Mask the prompt (set to -100) so the model doesn't learn to predict it
                for i in range(prefix_idx):
                    labels[i] = -100
                all_labels.append(labels)
            
            tokenized["labels"] = all_labels
            return tokenized

        # Apply tokenization and remove all raw columns (u_from, text,...)
        return dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=2,
            remove_columns=dataset.column_names if not streaming else None,
            desc="Tokenizing and masking evaluation data"
        )
    else:
        # FOR TRAINING: Return raw text (SFTTrainer + Collator will handle masking)
        dataset = dataset.map(format_batch_fn, **map_kwargs)
        
        # Cleanup: Remove u_from, a_text, etc. Keep ONLY 'text'
        cols_to_remove = [c for c in dataset.column_names if c != "text"]
        dataset = dataset.remove_columns(cols_to_remove)
        
        return dataset

class ClearCacheCallback(TrainerCallback):
    """
    Cleans up GPU memory after ... steps/epochs
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n--- GPU Cache Cleared after Epoch {state.epoch} ---")

    def on_step_end(self, args, state, control, **kwargs):
        # Optional: Clear every ... steps if your model is very large
        if state.global_step % 30 == 0:
            gc.collect()
            torch.cuda.empty_cache()

# This class handles the Validation Process
class LoRAMonitorCallback(TrainerCallback):
    def __init__(self, eval_loader, writer, local_rank, check_every, max_batches):
        # Store the variables we need inside the class
        self.eval_loader = eval_loader
        self.writer = writer
        self.local_rank = local_rank
        self.check_every = check_every
        self.max_batches = max_batches

    def on_step_end(self, args, state, control, **kwargs):
        
        if state.global_step > 0 and state.global_step % self.check_every == 0:
            # We call the monitor function
            # kwargs['model'] gives us the current state of the model
            monitor_lora_effectiveness(
                model=kwargs['model'], 
                eval_loader=self.eval_loader, 
                step=state.global_step, 
                writer=self.writer, 
                local_rank=self.local_rank,
                max_batches=self.max_batches
            )

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        save_every: int,
        output_dir: str = "./outputs",
        data_collator: Any = None,
        callbacks: list = None,
        hub_token: str = None,
        hub_model_id: str = None,
    ) -> None:
        """
        Initializes the SFTTrainer wrapper for Unsloth fine-tuning.
    
        """
        
        # Identify DDP ranks from environment variables
        
        self.local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank: int = int(os.environ.get("RANK", 0))

        self.model = model.module if hasattr(model, "module") else model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.save_every = save_every

        # Setup Training Configuration
        self.training_args = TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=50,
            gradient_checkpointing="unsloth",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters = False,
            group_by_length = True,
            dataloader_num_workers=1,      # Set this to 2 or 4 on Kaggle
            dataloader_prefetch_factor=2,

            dataloader_drop_last=True,
            
            warmup_steps=1,
            max_steps=120,
            save_steps=self.save_every,
            learning_rate=2e-4,
            remove_unused_columns = True,
            
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            #ddp_find_unused_parameters = False,
            
            logging_dir = "./logs",
            logging_steps=30,
            output_dir=output_dir,
            #disable_tqdm = True,

            push_to_hub= False,               # Enable auto-push
            hub_model_id="GeorgeNguyen/llama-3-groupchat-adapters",
            hub_token=hub_token,
            hub_strategy="checkpoint",

            report_to = "none", # Turn this on if using TensorBoard

            do_eval = False, # Don't turn this on, we have external function to handle Validation
            eval_strategy = "no", # Check validation every X steps
            eval_steps = None,
            
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3768,
            
        )

        # Initialize the SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            max_seq_length=512,
            args=self.training_args,
            packing = False,
            dataset_text_field="text",
            dataset_num_proc=2,
            data_collator=data_collator,
            callbacks = (callbacks or []), #[ClearCacheCallback()] +
        )

    def train(self) -> None:
        """Starts the training process using SFTTrainer."""
        
        if self.local_rank == 0:
            print(f"Starting training on Global Rank: {self.global_rank}")
            
        self.trainer.train()

    def save_model(self, path: str) -> None:
        """Saves the final LoRA adapters and tokenizer."""
        if self.local_rank == 0:
            self.trainer.save_model(path)
            print(f"Model saved to {path}")
            #self.trainer.push_to_hub()
            #print("pushed to HF from save_model function")

def load_train_objs(model_name: str, df_path, hf_token):

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 512,
        load_in_4bit = True,
        token = hf_token,          
        revision = "main",
        force_download = True, #I split dataset into parts, upload Lora to HF, then repull Adatpers for next Training Session.
        device_map = {"": torch.cuda.current_device()}
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )
    
    return model, tokenizer

def monitor_lora_effectiveness(model, eval_loader, step, writer, local_rank, max_batches):
    """
    Quiet validation that preserves the 'writer' for future use.
    If writer=None, it only prints to console.
    If writer is provided, it logs to TensorBoard as well.
    """
    model.eval()
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    torch.cuda.empty_cache()
    
    total_loss = torch.tensor(0.0).to(f"cuda:{local_rank}")
    count = torch.tensor(0.0).to(f"cuda:{local_rank}")
    
    with torch.inference_mode():
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
                
            inputs = {
                k: v.to(f"cuda:{local_rank}", non_blocking=True) if isinstance(v, torch.Tensor) 
                else v for k, v in batch.items()
            }
            
            outputs = model(**inputs)
            total_loss += outputs.loss.detach()
            count += 1
            del outputs
            del inputs
            # This is the "Nuclear Option" inside the loop:
            torch.cuda.empty_cache()
            
    # Sync across GPUs
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    
    if local_rank == 0:
        avg_loss = (total_loss / count).item()
        
        # 1. Console Output (The 'Quiet' part)
        print(f"[*] Step {step} | Val Loss: {avg_loss:.4f}")
        
        # 2. Future-Proof Logging (Only runs if you provide a writer)
        if writer is not None:
            writer.add_scalar("Effectiveness/Validation_Loss", avg_loss, step)
            writer.flush() 
    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()
    model.config.use_cache = True
    model.train()

def merge_and_save_final_model(adapter_path, output_dir, save_method):
    """
    Merges LoRA adapters into the base model and saves the result.
    """
    print(f"\n--- Merging LoRA into base model for UI deployment ---")
    
    try:
        # Load the model and tokenizer
        # We use load_in_4bit=False because merging usually requires 16-bit precision
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = adapter_path,
            max_seq_length = 512,
            load_in_4bit = False, 
        )
        
        # Switch to inference mode for stability
        FastLanguageModel.for_inference(model)
        
        # Prepare Output Directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Perform the Merge and Save
        model.save_pretrained_merged(
            output_dir, 
            tokenizer, 
            save_method = save_method
        )
        
        print(f"✅ Model merged and saved successfully to: {output_dir}")
        
        # Clean up memory immediately after merging
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_dir

    except Exception as e:
        print(f"❌ Merge failed! Error: {e}")
        return None

def main() -> None:
    
    # DDP Initialization
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Remove spam in log
    if dist.is_initialized() and dist.get_rank() != 0:
        import builtins
        builtins.print = lambda *args, **kwargs: None

    # Paths and Config
    dataset_path = '/kaggle/input/final-result/final_result.json'
    validation_path = '/kaggle/input/validation-export/validation-chat.json'
    selected_model = "unsloth/llama-3-8b-Instruct-bnb-4bit" # Use this in the first session
    #selected_model = "GeorgeNguyen/llama-3-groupchat-adapters" # Use this in the next sessions
    writer = SummaryWriter("/kaggle/working/logs") if local_rank == 0 else None
    my_repo = "GeorgeNguyen/llama-3-groupchat-adapters"
    my_secret_key = "HF_TOKEN"

    # Connect with Hugging Face
    hf_token = authenticate_hf(token_name=my_secret_key)
    # Load Objects & Data
    model, tokenizer = load_train_objs(model_name=selected_model, 
                                       df_path=dataset_path, 
                                       hf_token=hf_token)
    
    # Prepare both datasets
    train_dataset = prepare_hf_dataset(df_path=dataset_path, 
                                       tokenizer=tokenizer,
                                       skip_count=0, 
                                       is_tokenized=False)
    train_dataset = train_dataset.shuffle(seed=3768).shard(num_shards=2, index=local_rank)
    eval_dataset = prepare_hf_dataset(df_path=validation_path,
                                      tokenizer=tokenizer,
                                      skip_count=300, 
                                      is_tokenized=True)
    
    # Handle DDP Model wrapping if necessary
    trainable_model = model.module if hasattr(model, "module") else model  
    
    # Setup Custom Evaluation Loader
    # DistributedSampler ensures GPUs don't validate on the same data
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8, # Change this to optimize Validation Wait Time  
        num_workers=2,          # Use 2 workers per GPU
        pin_memory=True,        # Speeds up CPU-to-GPU transfer
        persistent_workers=True, 
        sampler=eval_sampler,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    # Initialize Callback and Trainer
    my_monitor = LoRAMonitorCallback(
        eval_loader=eval_loader, 
        writer=writer, 
        local_rank=local_rank, 
        check_every=50,
        max_batches=6,
    )
    
    response_template = [68431, 68883, 26289, 68431]
    instruction_template = [69784, 6584, 68431]
    
    collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    instruction_template=instruction_template,
    tokenizer=tokenizer
    )
    
    custom_trainer = Trainer(
        model = trainable_model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = None,
        output_dir = "./unsloth_ddp_checkpoints",
        hub_token=hf_token,
        save_every=100,
        callbacks=[my_monitor], #ClearCacheCallback()
        data_collator=collator,
    )

    # Keep track of Training Sessions, Steps/Session
    session_index = 1
    last_step = 250*(session_index-1)
    current_step = last_step + custom_trainer.training_args.max_steps

    # Start Training
    if local_rank == 0:
        print(f"--- Starting Distributed Training on {dist.get_world_size()} GPUs ---")
    
    custom_trainer.train()

    # Saving the Lora Adapter
    if local_rank == 0:
        print("Training complete. Saving final LoRA adapters...")
        custom_trainer.save_model("./final_lora_adapter")

        print(f"--- Preparing to push Step {current_step} to Hugging Face ---")   
        
        merged_path = merge_and_save_final_model(adapter_path="./final_lora_adapter", 
                                           output_dir="final_merged_model", 
                                           save_method="merged_16bit")

        api = HfApi(token=hf_token) # Connection opens here

        final_repo = "GeorgeNguyen/llama-3-groupchat-final"

        # Force create the repos (if they exist, exist_ok=True prevents errors)
        
        create_repo(repo_id=final_repo, token=hf_token, exist_ok=True)
        
        # Pushing separate Training Sessions, turn this off in the last Session
        # api.upload_folder(
        #     folder_path="./final_lora_adapter",
        #     repo_id=repo_id,
        #     commit_message=f"Session {session_index} complete, finish training step {current_step}",
        # )

        api.upload_folder(
            folder_path="final_merged_model",
            repo_id="GeorgeNguyen/llama-3-groupchat-final", # Dedicated repo for the UI
            commit_message="Production ready merged model",
        )
        
        print(f"Successfully pushed to Hugging Face")
        dist.destroy_process_group()
    else:
        print("Secondary process exiting. Only Rank 0 uploads.")

if __name__ == "__main__":
    main()
