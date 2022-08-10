import pdfplumber

pdf = pdfplumber.open('test.pdf')
page = pdf.pages[0]
text = page.extract_text()
# text = text.replace("\uf70a","่")
# text = text.replace("\uf70b","้")
# text = text.replace("\uf70e","์")
# text = text.replace("\uf712", "็")
# text = text.replace("\uf20c", "eeee")
# text = text.replace("\u201c","ffff")
# text = text.replace("\u201d","็")
# text = text.replace("\uf701","hhhh")
# text = text.replace("\uf702","ี")
print(text)