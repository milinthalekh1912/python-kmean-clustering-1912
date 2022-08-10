#
import pdfplumber
import pandas as pd

pdfRead = pdfplumber.open("8379.pdf")
table = pdfRead.pages[13].extract_table()
df = pd.DataFrame(table[1::], columns=table[0])

writer = pd.ExcelWriter('8379.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()
