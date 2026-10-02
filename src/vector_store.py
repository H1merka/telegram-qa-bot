from langchain_community.vectorstores import FAISS
from src.settings import DIM
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


class VecStore:
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.store = None

    def make_store(self):
        try:
            self.store = FAISS.load_local(
                self.path, 
                self.model, 
                allow_dangerous_deserialization=True
            )
        except RuntimeError:
            index = faiss.IndexFlatL2(DIM)
            self.store = FAISS(
                embedding_function=self.model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            self.store.save_local(self.path)

    def load_data(self, data):
        self.store.add_documents(data)
        self.store.save_local(self.path)

    def make_retriever(self, top_k=4):
        return self.store.as_retriever(
            search_kwargs={"k": top_k}
        )
