import pdfplumber

class PDFReader:
    @staticmethod
    def read_pdf(file_path):
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
