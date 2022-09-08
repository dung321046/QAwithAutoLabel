import torch
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("./temp")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

import pickle

with open("tokenized_squad.pk", "rb") as fr:
    tokenized_squad = pickle.load(fr)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from datasets import load_dataset

squad = load_dataset("squad")
from transformers import pipeline

nlp = pipeline('question-answering', model="./temp", tokenizer="./temp")
import xlsxwriter

workbook = xlsxwriter.Workbook('squad_sample_01.xlsx')
worksheet = workbook.add_worksheet()
worksheet.set_column(0, 0, 400)
worksheet.set_column(1, 1, 100)
context_format = workbook.add_format({'text_wrap': True})

question_format = workbook.add_format({'text_wrap': True})
row = 0
header = ["Context", "Question", "Human Answer", "Machine Answer", "Prob", "Start", "End"]
for c, t in enumerate(header):
    worksheet.write(row, c, header[c])
inputs = []
for i in range(10):
    inputs.append({'question': squad["train"][i]["question"],
                   'context': squad["train"][i]["context"]})
# inputs = squad["train"][:10]
predicts = nlp(inputs)
print(predicts)
print("----", predicts[0])

for i in range(10):
    row = i + 1
    worksheet.write(row, 0, squad["train"][i]["context"], context_format)
    worksheet.write(row, 1, squad["train"][i]["question"], question_format)
    worksheet.write(row, 3, predicts[i]['answer'])
    worksheet.write(row, 4, predicts[i]['score'])
    worksheet.write(row, 5, predicts[i]['start'])
    worksheet.write(row, 6, predicts[i]['end'])
workbook.close()
