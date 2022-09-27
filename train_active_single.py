import pickle
import random

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, TrainingArguments
from transformers import AutoTokenizer
from transformers import DefaultDataCollator

import wandb
from qa_trainer import QATrainer
from utils import get_accuracy

model_name = "srcocotero/tiny-bert-qa"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
data_collator = DefaultDataCollator()

tokenizer = AutoTokenizer.from_pretrained(model_name)
with open("tokenized_squad.pk", "rb") as fr:
    tokenized_squad = pickle.load(fr)
learning_rate = 5e-5
num_train_epochs = 100
weight_decay = 0.00
seed_number = 17
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
)
n = len(tokenized_squad["train"])
np.random.seed(seed_number)
random.seed(seed_number)
number_of_labels = 1000
sample_idx = random.sample(range(n), number_of_labels)
subtrain = tokenized_squad["train"].select(sample_idx)
m = len(tokenized_squad["validation"])
sample_test_idx = random.sample(range(m), 200)
valid = tokenized_squad["validation"].select(sample_test_idx)
trainer = QATrainer(
    model=model,
    args=training_args,
    train_dataset=subtrain,
    eval_dataset=valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
model.cuda()
valid_inputs = torch.tensor(valid["input_ids"], dtype=torch.int).cuda()
valid_att_mask = torch.tensor(valid["attention_mask"]).cuda()
predict = model(valid_inputs, attention_mask=valid_att_mask)
s_acc, e_acc, acc = get_accuracy(valid, predict)

wandb.login(key="17cc4e6449d9bc429d81a90211f21adf938bd629")
wandb.init()
run_name = wandb.run.name
wandb.define_metric("valid_step")
wandb.define_metric("Valid sAcc", step_metric="valid_step")
wandb.define_metric("Valid eAcc", step_metric="valid_step")
wandb.define_metric("Valid Acc", step_metric="valid_step")
wandb.define_metric("Cost", step_metric="valid_step")
wandb.log({"weight_decay": weight_decay, "learning_rate": learning_rate, "num_train_epochs": num_train_epochs,
           "seed_number": seed_number})
wandb.log({"Valid sAcc": s_acc, "Valid eAcc": e_acc,
           "Cost": 0,
           "Valid Acc": acc,
           "valid_step": 0})
for global_step in range(10):
    trainer.train()
    predict = model(valid_inputs, attention_mask=valid_att_mask)
    s_acc, e_acc, acc = get_accuracy(valid, predict)
    wandb.log({"Valid sAcc": s_acc, "Valid eAcc": e_acc,
               "Cost": number_of_labels,
               "Valid Acc": acc,
               "valid_step": global_step + 1})
trainer.save_model(model_name + "-" + run_name)
