import argparse
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


parser = argparse.ArgumentParser("Active Learning Arguments")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=1000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=200, help="number of queries per round")
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
train_idx = np.random.choice(remaining_idx, size=args.n_init_labeled, replace=False)
remaining_idx = list(set(remaining_idx) - set(train_idx))
trainer = QATrainer(
    model=model,
    args=training_args,
    train_dataset=all_data.select(train_idx),
    eval_dataset=all_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
model.cuda()
acc_stat, entropies = inference(model, all_data, range(N))
if __name__ == "__main__":
    wandb.login(key="17cc4e6449d9bc429d81a90211f21adf938bd629")
    wandb.init(project="QAwithAutoLabel", entity="henry93")
    run_name = wandb.run.name
    wandb.define_metric("valid_step")
    wandb.define_metric("Valid sAcc", step_metric="valid_step")
    wandb.define_metric("Valid eAcc", step_metric="valid_step")
    wandb.define_metric("Valid Acc", step_metric="valid_step")
    wandb.define_metric("Cost", step_metric="valid_step")
    wandb.log({"weight_decay": weight_decay, "learning_rate": learning_rate, "num_train_epochs": num_train_epochs,
               "seed_number": seed_number})
    wandb.log({"Valid sAcc": acc_stat[0], "Valid eAcc": acc_stat[1], "Valid Acc": acc_stat[2],
               "Cost": len(train_idx), "valid_step": 0})
    for global_step in range(args.n_round + 1):
        # Inference all data
        acc_stat, entropies = inference(model, all_data, range(N))
        # Logging acc of all data
        wandb.log({"Valid sAcc": acc_stat[0], "Valid eAcc": acc_stat[1], "Valid Acc": acc_stat[2],
                   "Cost": len(train_idx), "valid_step": global_step + 1})
        # We records the accuracy of the last step and exit
        if global_step == args.n_round:
            break
        # Select training sample
        if args.strategy_name == "RandomSampling":
            new_idx = np.random.choice(remaining_idx, size=args.n_query, replace=False)
        else:
            sorted_idx = np.argsort(entropies)
            new_idx = []
            for id in sorted_idx[::-1]:
                if id in remaining_idx:
                    new_idx.append(id)
                    if len(new_idx) == args.n_query:
                        break
        train_idx = np.concatenate([train_idx, new_idx])
        remaining_idx = list(set(remaining_idx) - set(train_idx))
        trainer = QATrainer(model=model, args=training_args, train_dataset=all_data.select(train_idx),
                            eval_dataset=all_data, tokenizer=tokenizer, data_collator=data_collator)
        trainer.train()

    trainer.save_model(model_name + "-" + run_name)
