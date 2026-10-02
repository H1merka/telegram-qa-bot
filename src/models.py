from langchain_huggingface import (
    HuggingFacePipeline,
    HuggingFaceEmbeddings,
    ChatHuggingFace
)


class Model:
    def __init__(self, name, path, device, batch_size, token):
        self.name = name
        self.path = path
        self.device = device
        self.batch_size = batch_size
        self.token = token


class ChatModel(Model):
    def __init__(
            self, name, path,
            device, batch_size,
            token, task, max_tokens,
            temperature, top_p,
            repetition_penalty,
            quant_config
        ):
        super().__init__(
            name, path, 
            device, batch_size, 
            token
        )
        self.task = task
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetiton_penalty = repetition_penalty
        self.quant_config = quant_config

    def chat_instance(self):
        llm = HuggingFacePipeline.from_model_id(
            model_id=self.name,
            task=self.task,
            pipeline_kwargs={
                "max_new_tokens": self.max_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetiton_penalty
            },
            device=self.device,
            batch_size=self.batch_size,
            model_kwargs={
                "cache_dir": self.path,
                "quantization_config": self.quant_config,
                "trust_remote_code": True,
                "token": self.token,
            }
        )

        return ChatHuggingFace(llm=llm)


class EmbeddingModel(Model):
    def __init__(
            self, name, path,
            device, batch_size,
            token, 
        ):
        super().__init__(
            name, path, 
            device, batch_size, 
            token
        )
    
    def embedding_instance(self):
        embeddings_model = HuggingFaceEmbeddings(
            model_name=self.name,
            cache_folder=self.path,
            encode_kwargs={
                "batch_size": self.batch_size,
                "normalize_embeddings": True
            },
            model_kwargs={
                "token": self.token,
                "device": self.device
            },
            show_progress=True
        )

        return embeddings_model
