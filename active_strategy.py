import math

import numpy as np
import torch


def get_entropy_from_logit(logits):
    from scipy.stats import entropy
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return [entropy(probs[i], base=2) for i in range(len(probs))]


def get_entropy(predict):
    start_entropies = get_entropy_from_logit(predict["start_logits"])
    end_entropies = get_entropy_from_logit(predict["end_logits"])
    return np.mean(np.asarray([start_entropies, end_entropies]), axis=0)


def get_highest_entropy(predict, topk):
    start_entropies = get_entropy_from_logit(predict["start_logits"])
    end_entropies = get_entropy_from_logit(predict["end_logits"])
    mean_entropies = np.mean(np.asarray([start_entropies, end_entropies]), axis=0)
    sorted_idx = np.argsort(mean_entropies)
    return sorted_idx[-topk:]



if __name__ == "__main__":
    import pickle
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    tokenizer = AutoTokenizer.from_pretrained("./srcocotero/tiny-bert-qa_run00")
    model = AutoModelForQuestionAnswering.from_pretrained("./srcocotero/tiny-bert-qa_run00")

    with open("tokenized_squad.pk", "rb") as fr:
        tokenized_squad = pickle.load(fr)
    idx = range(1000)
    subtrain = tokenized_squad["train"].select(idx)
    predict = model(torch.tensor(subtrain["input_ids"], dtype=torch.int),
                    attention_mask=torch.tensor(subtrain["attention_mask"]))
    output = model.bert(torch.tensor(subtrain["input_ids"], dtype=torch.int),
                        attention_mask=torch.tensor(subtrain["attention_mask"]))
    print(output["last_hidden_state"])
    print("Embedding:", output["last_hidden_state"][:, 0, :])
    # start_acc, end_acc, o_acc = get_accuracy(subtrain, predict)
    # print("Start-End-Overall Accuracy", start_acc, end_acc, o_acc)
    print(get_highest_entropy(predict, 10))
