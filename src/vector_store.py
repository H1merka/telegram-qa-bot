from langchain_community.vectorstores import FAISS
from src.settings import DIM, STORE_LOG_PATH
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from src.logger import AppLogger


logger = AppLogger(__name__, STORE_LOG_PATH)


class VecStore:
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.store = None
        self.logger = logger
        self.logger.debug("Store object inintialized")

    def make_store(self):
        self.logger.debug("Start store making")
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
        self.logger.debug("Store made")

    def load_data(self, data):
        self.logger.debug("Start adding data")
        self.store.add_documents(data)
        self.store.save_local(self.path)
        self.logger.debug("Data added")

    def make_retriever(self, top_k=4):
        self.logger.debug("Start retriever making")
        retriever = self.store.as_retriever(
            search_kwargs={"k": top_k}
        )
        self.logger.debug("Retriever made")
        return retriever
