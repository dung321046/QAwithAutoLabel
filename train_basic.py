import math
import pickle
import random

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, TrainingArguments
from transformers import AutoTokenizer
from transformers import DefaultDataCollator

import wandb
from qa_trainer import QATrainer
from utils import get_num_corrects


def get_entropy_from_logit(logits):
    from scipy.stats import entropy
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return [entropy(probs[i], base=2) for i in range(len(probs))]


def get_entropy(predict):
    start_entropies = get_entropy_from_logit(predict["start_logits"])
    end_entropies = get_entropy_from_logit(predict["end_logits"])
    return np.mean(np.asarray([start_entropies, end_entropies]), axis=0)


def inference(model, data, idx):
    num = len(idx)
    batch_size = 16
    num_batch = int(math.ceil(1.0 * num / batch_size))
    entropies = []
    acc_stat = np.zeros(3)
    for batch_idx in range(num_batch):
        sub_idx = idx[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
        sub_input = data.select(sub_idx)
        with torch.no_grad():
            sub_predict = model(torch.tensor(sub_input["input_ids"], dtype=torch.int).cuda(),
                                attention_mask=torch.tensor(sub_input["attention_mask"]).cuda())
            # sub_last_layer = model.bert(torch.tensor(sub_input["input_ids"], dtype=torch.int).cuda(),
            #                     attention_mask=torch.tensor(sub_input["attention_mask"]).cuda())
            sub_entropy = get_entropy(sub_predict)
            entropies.extend(sub_entropy)
            b_acc_stat = get_num_corrects(sub_input, sub_predict)
            acc_stat = [sum(i) for i in zip(acc_stat, b_acc_stat)]
    acc_stat = [cor / num for cor in acc_stat]
    return acc_stat, entropies


model_name = "srcocotero/tiny-bert-qa"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
data_collator = DefaultDataCollator()

tokenizer = AutoTokenizer.from_pretrained(model_name)
with open("tokenized_squad.pk", "rb") as fr:
    tokenized_squad = pickle.load(fr)
learning_rate = 5e-5
num_train_epochs = 20
weight_decay = 0.00
seed_number = 17
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
)

n = len(tokenized_squad["train"])
np.random.seed(seed_number)
random.seed(seed_number)

number_of_data = 10000
sample_data = random.sample(range(n), number_of_data)
all_data = tokenized_squad["train"].select(sample_data)
number_of_labels = 5000
sample_train = random.sample(range(number_of_data), number_of_labels)
train_data = all_data.select(sample_train)

model.cuda()
acc_stat, entropies = inference(model, all_data, range(number_of_data))

wandb.login(key="17cc4e6449d9bc429d81a90211f21adf938bd629")
wandb.init(project="QAwithAutoLabel", entity="henry93")
run_name = wandb.run.name
wandb.log({"weight_decay": weight_decay, "learning_rate": learning_rate, "num_train_epochs": num_train_epochs,
           "seed_number": seed_number, "num_data": number_of_data})
wandb.define_metric("valid_step")
wandb.define_metric("Valid sAcc", step_metric="valid_step")
wandb.define_metric("Valid eAcc", step_metric="valid_step")
wandb.define_metric("Valid Acc", step_metric="valid_step")
wandb.define_metric("Cost", step_metric="valid_step")
wandb.log({"weight_decay": weight_decay, "learning_rate": learning_rate, "num_train_epochs": num_train_epochs,
           "seed_number": seed_number})
wandb.log({"Valid sAcc": acc_stat[0], "Valid eAcc": acc_stat[1], "Valid Acc": acc_stat[2],
           "Cost": number_of_labels, "valid_step": 0})
for global_step in range(20):
    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=all_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    # Inference all data
    acc_stat, entropies = inference(model, all_data, range(number_of_data))
    # Logging acc of all data
    wandb.log({"Valid sAcc": acc_stat[0], "Valid eAcc": acc_stat[1], "Valid Acc": acc_stat[2],
               "Cost": number_of_labels, "valid_step": global_step + 1})
    print("Valid sAcc", acc_stat[0], "Valid eAcc", acc_stat[1], "Valid Acc", acc_stat[2], "valid_step", global_step + 1)
trainer.save_model(model_name + "-" + run_name)
