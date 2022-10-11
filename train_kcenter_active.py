import argparse
import math
import pickle
import random

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from transformers import AutoModelForQuestionAnswering, TrainingArguments
from transformers import AutoTokenizer
from transformers import DefaultDataCollator

import wandb
from qa_trainer import QATrainer
from utils import get_num_corrects


class kCenterGreedy:
    def __init__(self, features, metric='euclidean'):
        self.features = features
        self.name = 'kcenter'
        self.metric = metric
        self.min_distances = None

    def update_distances(self, cluster_centers, only_new=True):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        # Update min_distances for all examples given new cluster center.
        x = self.features[cluster_centers]
        dist = pairwise_distances(self.features, x, metric=self.metric)
        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1)
        else:
            self.min_distances = np.minimum(self.min_distances, np.squeeze(dist))

    def select_batch_(self, features, already_selected, N):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """
        self.already_selected = already_selected

        print('Getting transformed features...')
        self.features = features
        print('Calculating distances...')
        self.update_distances(already_selected, only_new=False)
        new_batch = []
        for _ in range(N):
            ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected
            self.update_distances([ind], only_new=True)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch


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
    embeddings = []
    acc_stat = np.zeros(3)
    for batch_idx in range(num_batch):
        sub_idx = idx[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
        sub_input = data.select(sub_idx)
        with torch.no_grad():
            sub_predict = model(torch.tensor(sub_input["input_ids"], dtype=torch.int).cuda(),
                                attention_mask=torch.tensor(sub_input["attention_mask"]).cuda())
            sub_last_layer = model.bert(torch.tensor(sub_input["input_ids"], dtype=torch.int).cuda(),
                                        attention_mask=torch.tensor(sub_input["attention_mask"]).cuda())
            embeddings.extend(sub_last_layer['last_hidden_state'][:, 0, :].detach().cpu().numpy())
            b_acc_stat = get_num_corrects(sub_input, sub_predict)
            acc_stat = [sum(i) for i in zip(acc_stat, b_acc_stat)]
    acc_stat = [cor / num for cor in acc_stat]
    embeddings = np.asarray(embeddings)
    return acc_stat, embeddings


parser = argparse.ArgumentParser("Active Learning Arguments")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=1000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=200, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=20, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="IMDB", choices=["IMDB"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="KCenterGreedy",
                    choices=["RandomSampling",
                             "KCenterGreedy",
                             ], help="query strategy")

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
    output_dir="./results2",
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
number_of_data = 10000
all_idx = random.sample(range(n), number_of_data)
all_data = tokenized_squad["train"].select(all_idx)

remaining_idx = range(number_of_data)
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
acc_stat, embeddings = inference(model, all_data, range(number_of_data))
trainer.train()
model.cuda()
acc_stat, embeddings = inference(model, all_data, range(number_of_data))
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
               "seed_number": seed_number, "num_data": number_of_data, "strategy": args.strategy_name})
    wandb.log({"Valid sAcc": acc_stat[0], "Valid eAcc": acc_stat[1], "Valid Acc": acc_stat[2],
               "Cost": len(train_idx), "valid_step": 0})
    for global_step in range(args.n_round + 1):
        # Inference all data
        acc_stat, embeddings = inference(model, all_data, range(number_of_data))
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
            coreset = kCenterGreedy(embeddings)
            new_idx = coreset.select_batch_(embeddings, train_idx, args.n_query)
        train_idx = np.concatenate([train_idx, new_idx])
        remaining_idx = list(set(remaining_idx) - set(train_idx))
        trainer = QATrainer(model=model, args=training_args, train_dataset=all_data.select(train_idx),
                            eval_dataset=all_data, tokenizer=tokenizer, data_collator=data_collator)
        trainer.train()

    trainer.save_model(model_name + "-" + run_name)
