import pickle

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("./srcocotero/tiny-bert-qa_run00")
model = AutoModelForQuestionAnswering.from_pretrained("./srcocotero/tiny-bert-qa_run00")

with open("tokenized_squad.pk", "rb") as fr:
    tokenized_squad = pickle.load(fr)
idx = range(100)
subtrain = tokenized_squad["train"].select(idx)
print(subtrain)
predict = model(torch.tensor(subtrain["input_ids"], dtype=torch.int),
                attention_mask=torch.tensor(subtrain["attention_mask"]))
print(predict)
start_pred = torch.argmax(predict["start_logits"], dim=1)
end_pred = torch.argmax(predict["end_logits"], dim=1)
# subtrain["start_positions"]
n = len(subtrain["start_positions"])
start_acc, end_acc = np.zeros(n), np.zeros(n)
for i in range(n):
    if start_pred[i] == subtrain["start_positions"][i]:
        start_acc[i] = 1
    if end_pred[i] == subtrain["end_positions"][i]:
        end_acc[i] = 1
print("Acc start", sum(start_acc) / n)
print("Acc end", sum(end_acc) / n)
accs = [end_acc[i] and start_acc[i] for i in range(n)]
print("Valid Acc", sum(accs) / n)
# end_pred = torch.argmax(end_scores, dim=1)
# print(start_pred)
# print(subtrain["start_ids"])
# print(end_pred)
# all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
# answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
# answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
