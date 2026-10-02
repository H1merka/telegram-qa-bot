from src.utils import format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.settings import SYSTEM_PROMPT


class Chain:
    def __init__(
            self, 
            preprocessor, 
            vector_store, 
            chat_model, 
        ):
        self.preprocessor = preprocessor
        self.vector_store = vector_store
        self.chat_model = chat_model

    def load_data(self, data, file_name=None):
        self.preprocessor.preprocess(file_name, data)
        self.vector_store.load_data(self.preprocessor.get_content)

    def invoke(self, question, prompt=SYSTEM_PROMPT, top_k=4):
        retriever = self.vector_store.make_retriever(top_k)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        rag_chain = (
            {
                "context": retriever | format_docs,  # Retrieve docs and format them
                "input": RunnablePassthrough()       # Pass the question through
            }
            | prompt_template                        # Format the prompt with context and question
            | self.chat_model                                   # Send to language model
            | StrOutputParser()                     # Parse the output to a string
        )

        invocation = rag_chain.invoke(question)
        idx = invocation.find("<|im_start|>assistant")
        if idx == -1:
            return invocation
        else:
            return invocation[idx+21:]
