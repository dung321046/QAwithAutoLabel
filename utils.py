import xlsxwriter


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
