from pypdf import PdfReader
from docx import Document as Doc
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils import clean_extracted_text

class Preprocessor:
    def __init__(
            self, 
            chunk_size, 
            chunk_overlap,
            length_func=len,
            separators=None,
            file_name=None,
            content=None):
        self.file_name = file_name
        self.content = content
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_func = length_func
        self.separators = separators

    @property
    def get_file_name(self):
        return self.file_name

    @property
    def get_content(self):
        return self.content

    def _text_split(self):
        if not self.separators:
            self.separators = ["\n\n", "\n", ". ", " ", ""]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_func,
            separators=self.separators
        )
        self.content = text_splitter.split_text(self.content)

    def _document_make(self):
        documents = []
        for i, chunk in enumerate(self.content):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "chunk_length": len(chunk),
                    "source": "pdf_document"
                }
            )
            documents.append(doc)
        self.content = documents

    def _pdf_preprocess(self):
        pdf_reader = PdfReader(self.content)

        raw_text = ""
        for _, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            raw_text += page_text

        self.content = raw_text
        self.content = clean_extracted_text(self.content)
        self._text_split()
        self._document_make()

    def _txt_preprocess(self):
        self.content = self.content.read().decode('utf-8')
        self.content = clean_extracted_text(self.content)
        self._text_split()
        self._document_make()    

    def _docx_preprocess(self):
        doc = Doc(self.content)
        
        self.content = '\n'.join([p.text for p in doc.paragraphs])
        self.content = clean_extracted_text(self.content)
        self._text_split()
        self._document_make()

    def _plain_preprocess(self):
        self.content = clean_extracted_text(self.content)
        self._text_split()
        self._document_make()    

    def preprocess(self, file_name, content):
        self.file_name = file_name
        self.content = content
        if self.file_name:
            if self.file_name.endswith(".pdf"):
                self._pdf_preprocess()
            elif self.file_name.endswith(".txt"):
                self._txt_preprocess()
            elif self.file_name.endswith(".doc") or self.file_name.endswith(".docx"):
                self._docx_preprocess()
        else:
            self._plain_preprocess()
    