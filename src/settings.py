from dotenv import load_dotenv
import os
import logging
import torch
from transformers import BitsAndBytesConfig


# Sensitive data
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
TELEGRAM_BOT_KEY = os.getenv("TELEGRAM_BOT_KEY")

# Keyboard
START_BUTTON = ["/start", "Старт"]
START_STR = START_BUTTON[-1]
HELP_BUTTON = ["/help", "Помощь"]
HELP_STR = HELP_BUTTON[-1]
ADD_DATA_BUTTON = ["/add_data", "Добавить данные"]
ADD_DATA_STR = ADD_DATA_BUTTON[-1]
CANCEL_BUTTON = ["/cancel", "Отмена"]
CANCEL_STR = CANCEL_BUTTON[-1]
ADDING_DATA = 1

# Model info
CHUNK_OVERLAP = 200
CHUNK_SIZE = 1000
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CHAT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_BATCH_SIZE = 64
CHAT_BATCH_SIZE = 4
MODEL_TASK = "text-generation"
MAX_NEW_TOKENS =  512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
DIM = 1024

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Log info
LOG_LEVEL = logging.DEBUG

# Paths
EMBEDDING_MODEL_PATH = "./model/embeddings"
CHAT_MODEL_PATH = "./model/chat"
DB_PATH = "./db"
LOG_PATH = "./logs"

SYSTEM_PROMPT = """
Ты — полезный ИИ-помощник, отвечающий на вопросы на основе предоставленного контекста.

Правила:

1. Для ответов на вопросы используй только информацию из предоставленного контекста.
2. Если в контексте недостаточно информации, прямо скажи об этом.
3. Отвечай конкретно и ссылайся на соответствующие фрагменты контекста.
4. Излагай мысли ясно и кратко.
5. Если ты не уверен, признайся в этом, а не пытайся угадать.

Контекст:
{context}

Вопрос: {input}

Отвечай на основе контекста выше:

"""