from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset
from random import sample
from pathlib import Path
import yaml
import argparse

from babylm_dataset import BabylmDataset



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config/llama-16M.yaml", help="Configuration file path")
parser.add_argument("--lr", type=float, default=None, help="Learning rate")
parser.add_argument("--model_name", type=str, default=None, help="Model name")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


# Override config parameters if provided as command-line arguments
if args.lr:
    config['training']['lr'] = args.lr
if args.model_name:
    config['model']['name'] = args.model_name


SEQ_LENGTH = config['data']['seq_length']

tokenizer_path = config['data']['tokenizer_path']
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(config['data']['eval_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), config['data']['eval_samples'])
eval_dataset = Subset(full_eval_dataset, eval_indices)

# We tokenize the whole dataset and then set the max length
tokenizer.model_max_length = SEQ_LENGTH

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Dynamic Model Configuration
if config['model']['type'] == "Llama":
    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config['model']['hidden_size'],
        intermediate_size=config['model']['intermediate_size'],
        num_hidden_layers=config['model']['n_layer'],
        num_attention_heads=config['model']['n_head'],
        # Add other parameters as needed
    )
    model = LlamaForCausalLM(model_config)
elif config['model']['type'] == "GPT2":
    model_config = GPT2Config(
        #TODO check the intermediate_size, I guess it is 4 * hidden
        vocab_size=tokenizer.vocab_size,
        n_positions=tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        # Add other parameters as needed
    )
    model = GPT2LMHeadModel(model_config)

print(f'model parameters = {model.num_parameters()}')


output_dir = Path(config['logging']['output_dir']) / config['model']['name']
accumulation_steps = config['training']['gradient_accumulation_steps']
per_device_bsz = config['training']['batch_size'] // accumulation_steps

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=4,
    gradient_accumulation_steps=accumulation_steps,
    per_device_train_batch_size=per_device_bsz,
    save_total_limit=1,  # Set to zero to avoid saving
    warmup_steps=config['training']['warmup_steps'], 
    lr_scheduler_type="cosine",
    learning_rate=float(config['training']['lr']),
    logging_steps=10,
    fp16=config['training']['fp16'],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


if __name__ == "__main__":

    if config['logging']['wandb']:
        import wandb
        wandb.login()
        wandb.init(project= config['logging']['project'], name=config['model']['name'], config=config)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)