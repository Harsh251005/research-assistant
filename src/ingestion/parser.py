import pymupdf

def extract_text_from_pdf(pdf_file_path):
    """Opens a PDF and extracts text from each page."""
    doc = pymupdf.open(pdf_file_path)

    page_content = []

    for page in doc:
        text = page.get_text()
        page_content.append(text)

    doc.close()

    return page_content
