import fitz
import os

class FileRetrieve():

    def __init__(self) -> None:
        self.folder_path = "pdf_files"

    def get_pdf_files(self):
        return [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path) if file.endswith('.pdf')]

    def extract_text_from_pdf(self, file_path):
        with fitz.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text

    def get_documents(self):
        pdf_files = self.get_pdf_files()
        return [self.extract_text_from_pdf(file) for file in pdf_files]

