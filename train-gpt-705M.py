from torch.utils.data import Subset
from transformers import GPT2TokenizerFast
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from random import sample
from pathlib import Path
import wandb


from babylm_dataset import BabylmDataset



#############
LR = 2.5e-4
BSZ = 256
SEQ_LENGTH = 128
#############

PATH = Path("./")
MODEL_NAME = 'gpt-705M'
MODEL_OUTPUT = Path('./models') /  MODEL_NAME
EVAL_SAMPLES = 8192


wandb_log = True


tokenizer_path = PATH / "models/gpt-clean-16000.json"

tokenizer = GPT2TokenizerFast(tokenizer_file = str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

print(len(train_dataset), len(eval_dataset))


tokenizer.model_max_length = SEQ_LENGTH


config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=tokenizer.model_max_length,
    n_embd=1536,
    n_layer=24,
    n_head=16,
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>")
)



model = GPT2LMHeadModel(config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

print(f'model num parameters = {model.num_parameters()}')



config_dict = config.to_dict()


if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME, config=config_dict)



training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=6,
    gradient_accumulation_steps=BSZ,
    per_device_train_batch_size=1,
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










