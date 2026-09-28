"""Generates table fixtures. Run from the project root:
    python tests/make_table_fixtures.py
"""
import csv

from docx import Document
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = "tests/fixtures"
ROWS = [["Model", "Company", "Year"], ["GPT-4", "OpenAI", "2023"], ["Claude 2", "Anthropic", "2023"]]
CAPTION = "Table 1: Model release dates."

# .csv
with open(f"{OUT}/models.csv", "w", newline="") as f:
    csv.writer(f).writerows(ROWS)

# .xlsx
wb = Workbook()
ws = wb.active
ws.title = "Models"
for r in ROWS:
    ws.append(r)
wb.save(f"{OUT}/models.xlsx")

# .docx: heading, paragraph, caption, table
d = Document()
d.add_heading("Company History", level=1)
d.add_paragraph("OpenAI was founded in 2015 in San Francisco and released GPT-4 in March 2023.")
d.add_paragraph(CAPTION)
t = d.add_table(rows=len(ROWS), cols=3)
for i, row in enumerate(ROWS):
    for j, value in enumerate(row):
        t.cell(i, j).text = value
d.save(f"{OUT}/report_with_table.docx")

# .md
with open(f"{OUT}/with_table.md", "w") as f:
    f.write(
        "# Company History\n"
        "OpenAI was founded in 2015 in San Francisco.\n\n"
        f"{CAPTION}\n\n"
        "| Model | Company | Year |\n"
        "|---|---|---|\n"
        "| GPT-4 | OpenAI | 2023 |\n"
        "| Claude 2 | Anthropic | 2023 |\n"
    )

# .pdf with a ruled table
styles = getSampleStyleSheet()
grid = Table(ROWS)
grid.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
SimpleDocTemplate(f"{OUT}/table.pdf", pagesize=A4).build([
    Paragraph("OpenAI released GPT-4 in March 2023 and it was widely covered.", styles["Normal"]),
    Spacer(1, 12),
    Paragraph(CAPTION, styles["Normal"]),
    Spacer(1, 6),
    grid,
])

# a text file pretending to be a Word document
with open(f"{OUT}/fake_docx.docx", "w") as f:
    f.write("just plain text, not a Word file")

print("Table fixtures created in", OUT)