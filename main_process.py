import os

import xlsxwriter
from datasets import load_dataset

DATA_PATH = "./data"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
LOG_FILE = DATA_PATH + "/log.txt"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        f.write("QA Data manager\n")


def write_data(inputs, file_index):
    file_name = DATA_PATH + "/qa_sample_" + str(file_index).zfill(4) + ".xlsx"

    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, 0, 300)
    worksheet.set_column(1, 1, 50)
    worksheet.set_column(2, 3, 20)
    context_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})

    question_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
    row = 0
    header = ["Context", "Question", "Human Answer", "Machine Answer", "Prob", "Start", "End", "Machine Version"]
    for c, t in enumerate(header):
        worksheet.write(row, c, header[c])
    for i, input in enumerate(inputs):
        row = i + 1
        worksheet.write(row, 0, input["context"], context_format)
        worksheet.write(row, 1, input["question"], question_format)
    workbook.close()


squad = load_dataset("squad")
print(squad["train"])
print(len(squad["train"]))
n = len(squad["train"])
batch_size = 50
start = 0

file_index = 0
while start < n:
    end = min(n, start + batch_size)
    inputs = []
    for i in range(start, end):
        inputs.append({'question': squad["train"][i]["question"],
                       'context': squad["train"][i]["context"]})
    write_data(inputs, file_index)
    start = end
    file_index += 1
