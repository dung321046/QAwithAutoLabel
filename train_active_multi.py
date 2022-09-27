import argparse
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

parser = argparse.ArgumentParser("Active Learning Arguments")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=20, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=20, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="IMDB", choices=["IMDB"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                    choices=["RandomSampling",
                             "LeastConfidence",
                             "MarginSampling",
                             "EntropySampling",
                             "LeastConfidenceDropout",
                             "MarginSamplingDropout",
                             "EntropySamplingDropout",
                             "KMeansSampling",
                             "KCenterGreedy",
                             "BALDDropout",
                             "AdversarialBIM",
                             "AdversarialDeepFool"], help="query strategy")

args = parser.parse_args()
model_name = "srcocotero/tiny-bert-qa"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
data_collator = DefaultDataCollator()

tokenizer = AutoTokenizer.from_pretrained(model_name)
with open("tokenized_squad.pk", "rb") as fr:
    tokenized_squad = pickle.load(fr)
learning_rate = 5e-5
num_train_epochs = 10
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
N = 10000
all_idx = random.sample(range(n), N)
all_data = tokenized_squad["train"].select(all_idx)
remaining_idx = range(N)
number_of_labels = 20
train_idx = np.random.choice(remaining_idx, size=args.n_init_labeled, replace=False)
remaining_idx = list(set(remaining_idx) - set(train_idx))
trainer = QATrainer(
    model=model,
    args=training_args,
    train_dataset=all_data.select(train_idx),
    tokenizer=tokenizer,
    data_collator=data_collator,
)
model.cuda()
valid_inputs = torch.tensor(all_data["input_ids"], dtype=torch.int).cuda()
valid_att_mask = torch.tensor(all_data["attention_mask"]).cuda()
predict = model(valid_inputs, attention_mask=valid_att_mask)
s_acc, e_acc, acc = get_accuracy(all_data, predict)

if __name__ == "__main__":
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
    for global_step in range(args.n_round + 1):
        # Inference all data
        predict = model(valid_inputs, attention_mask=valid_att_mask)
        s_acc, e_acc, acc = get_accuracy(all_data, predict)
        # Logging acc of all data
        wandb.log({"Valid sAcc": s_acc, "Valid eAcc": e_acc,
                   "Cost": number_of_labels,
                   "Valid Acc": acc,
                   "valid_step": global_step + 1})
        # We records the accuracy of the last step and exit
        if global_step == args.n_round:
            break
        # Select training sample
        if args.strategy_name == "RandomSampling":
            new_idx = np.random.choice(remaining_idx, size=args.n_query, replace=False)
        else:
            import active_strategy

            order_idx = active_strategy.get_highest_entropy(predict, N)[::-1]
            new_idx = []
            for id in order_idx:
                if id in remaining_idx:
                    new_idx.append(id)
                    if len(new_idx) == args.n_query:
                        break
        train_idx = np.concatenate([train_idx, new_idx])
        remaining_idx = list(set(remaining_idx) - set(train_idx))
        trainer = QATrainer(model=model, args=training_args, train_dataset=all_data.select(train_idx),
                            tokenizer=tokenizer, data_collator=data_collator)
        trainer.train()

    trainer.save_model(model_name + "-" + run_name)
