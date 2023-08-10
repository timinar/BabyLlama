from torch.utils.data import Dataset, Subset
from transformers import GPT2TokenizerFast
from transformers import LlamaForCausalLM, LlamaConfig

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from random import sample
from pathlib import Path
import wandb



from babylm_dataset import BabylmDataset



#############
LR = 3e-4
BSZ = 128
SEQ_LENGTH = 128
#############

PATH = Path("./")
MODEL_NAME = f'Llama-360M'
MODEL_OUTPUT = Path('./models') /  MODEL_NAME
EVAL_SAMPLES = 8192

# Llama-360M  needs more VRAM than GPT2-705M
GRADIENT_ACCUMULATION_STEPS = 8
PER_DEVICE_BSZ = BSZ // GRADIENT_ACCUMULATION_STEPS


wandb_log = True


tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)




tokenizer.model_max_length = SEQ_LENGTH


config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=1024,
    num_hidden_layers=24,
    intermediate_size=3072,
    num_attention_heads=8,
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    max_position_embeddings=2*SEQ_LENGTH,
)



model = LlamaForCausalLM(config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

print(f'model num parameters = {model.num_parameters()}')



config_dict = config.to_dict()


if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME, config=config_dict)



training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=4,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    per_device_train_batch_size=PER_DEVICE_BSZ,
    save_total_limit=1,  # Set to zero to avoid saving
    warmup_steps=200, 
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=10,
    fp16=True,
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

trainer.train()



trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)












