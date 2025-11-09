import PyPDF2 
def load_document(file):
    if hasattr(file, "read"):  # file-like object (uploaded)
        content = None
        if file.name.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(file)
            content = "\n".join(page.extract_text() for page in reader.pages)
        elif file.name.endswith(".txt"):
            content = file.read().decode("utf-8")
        elif file.name.endswith(".docx"):
            import docx
            doc = docx.Document(file)
            content = "\n".join(p.text for p in doc.paragraphs)
        return content
    else:  # path string
        # existing logic for file paths
        pass
