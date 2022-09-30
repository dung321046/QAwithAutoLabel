import math

import numpy as np
import torch

import wandb
from utils import get_num_corrects


def get_entropy_from_logit(logits):
    from scipy.stats import entropy
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return [entropy(probs[i], base=2) for i in range(len(probs))]


def get_entropy(predict):
    start_entropies = get_entropy_from_logit(predict["start_logits"])
    end_entropies = get_entropy_from_logit(predict["end_logits"])
    return np.mean(np.asarray([start_entropies, end_entropies]), axis=0)


def inference(model, data):
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
        print(sub_predict)
    acc_stat = [cor / num for cor in acc_stat]
    return acc_stat, entropies


if __name__ == "__main__":
    wandb.login(key="17cc4e6449d9bc429d81a90211f21adf938bd629")
    wandb.init(project="Performance-Testing", entity="henry93")
    import pickle
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    model_name = "srcocotero/tiny-bert-qa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    with open("tokenized_squad.pk", "rb") as fr:
        tokenized_squad = pickle.load(fr)
    num = 500
    idx = range(num)
    subtrain = tokenized_squad["train"].select(idx)
    import timeit

    # start = timeit.timeit()
    # predict = model(torch.tensor(subtrain["input_ids"], dtype=torch.int),
    #                 attention_mask=torch.tensor(subtrain["attention_mask"]))
    # end = timeit.timeit()
    # wandb.log({"Step": 0, "Time": end - start})
    #
    # start = timeit.timeit()
    # output = model.bert(torch.tensor(subtrain["input_ids"], dtype=torch.int),
    #                     attention_mask=torch.tensor(subtrain["attention_mask"]))
    # end = timeit.timeit()
    # wandb.log({"Step": 1, "Time": end - start})
    start = timeit.timeit()
    model.cuda()
    acc_stat, entropies = inference(model, tokenized_squad["train"])
    end = timeit.timeit()
    print("Accuracy:", acc_stat)
    wandb.log({"Step": 2, "Time": end - start})
