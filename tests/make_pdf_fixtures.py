"""Generates PDF test fixtures. Run from the project root:
    python tests/make_pdf_fixtures.py
"""
from reportlab.pdfgen import canvas

OUT = "tests/fixtures"

BODY = [
    "OpenAI released GPT-4 in March 2023. The company was co-founded",
    "by Sam Altman, who later became its chief executive. Microsoft has",
    "invested billions in the company since 2019.",
]


def simple_pdf(path, pages=1, footer=False, encrypt=None):
    c = canvas.Canvas(path, encrypt=encrypt)
    for n in range(1, pages + 1):
        y = 780
        if footer:
            c.drawString(72, y, "Acme Research Report")      # repeating header
            y -= 40
        for line in BODY:
            c.drawString(72, y, line)
            y -= 18
        c.drawString(72, y - 18, f"Page-specific note number {n}.")
        if footer:
            c.drawString(72, 40, "Confidential - Acme Corp")  # repeating footer
            c.drawString(300, 40, f"Page {n}")                 # page number
        c.showPage()
    c.save()


simple_pdf(f"{OUT}/simple.pdf", pages=1)
simple_pdf(f"{OUT}/repeated_footers.pdf", pages=4, footer=True)   # rule 12
simple_pdf(f"{OUT}/encrypted.pdf", pages=1, encrypt="secret")     # rule 5

# A PDF with no pages at all, to test empty-document handling (rule 38)
c = canvas.Canvas(f"{OUT}/blank.pdf")
c.showPage()
c.save()

print("PDF fixtures created in", OUT)
