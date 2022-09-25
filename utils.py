import numpy as np
import torch
import xlsxwriter


def get_accuracy(dataset, predict):
    start_pred = torch.argmax(predict["start_logits"], dim=1)
    end_pred = torch.argmax(predict["end_logits"], dim=1)
    n = len(dataset["start_positions"])
    start_acc, end_acc = np.zeros(n), np.zeros(n)
    for i in range(n):
        if start_pred[i] == dataset["start_positions"][i]:
            start_acc[i] = 1
        if end_pred[i] == dataset["end_positions"][i]:
            end_acc[i] = 1
    accs = [end_acc[i] * start_acc[i] for i in range(n)]
    return sum(start_acc) / n, sum(end_acc) / n, sum(accs) / n


def acc_report(file_name, report_data):
    workbook = xlsxwriter.Workbook(file_name + ".xlsx")
    worksheet = workbook.add_worksheet()
    row = 0
    header = ["File", "#Records", "%Label", "%Acc"]
    worksheet.set_column(0, 0, 50)
    context_format = workbook.add_format({'text_wrap': True})
    for c, t in enumerate(header):
        worksheet.write(row, c, header[c])
    for i, row_data in enumerate(report_data):
        row = i + 1
        worksheet.write(row, 0, row_data["File"], context_format)
        worksheet.write(row, 1, row_data['#Records'])
        worksheet.write(row, 2, row_data['%Label'])
        worksheet.write(row, 3, row_data['%Acc'])
    workbook.close()


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    inputs["id"] = examples["id"]
    inputs["context"] = examples["context"]
    inputs["questions"] = questions
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
