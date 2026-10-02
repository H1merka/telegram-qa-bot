from functools import partial
from src.settings import (
    TELEGRAM_BOT_KEY, CHUNK_SIZE, CHUNK_OVERLAP,
    DB_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH,
    EMBEDDING_BATCH_SIZE, HF_TOKEN, DEVICE,
    CHAT_MODEL_NAME, CHAT_MODEL_PATH, CHAT_BATCH_SIZE,
    MODEL_TASK, MAX_NEW_TOKENS, TEMPERATURE,
    TOP_P, REPETITION_PENALTY, QUANTIZATION_CONFIG,
    ADD_DATA_BUTTON, START_BUTTON, HELP_BUTTON, CANCEL_BUTTON,
    ADDING_DATA
)
from src.chain import Chain
from src.preprocess import Preprocessor
from src.vector_store import VecStore
from src.models import ChatModel, EmbeddingModel
from src.handlers import (
    start, help, add_data_command,
    add_data, cancel, question_answer,
    fallback
)
from telegram.ext import (
    Application, ConversationHandler,
    MessageHandler, filters
)


def main():
    application = Application.builder().token(TELEGRAM_BOT_KEY).build()
    
    preprocessor = Preprocessor(CHUNK_SIZE, CHUNK_OVERLAP)
    embedding_model = EmbeddingModel(
        EMBEDDING_MODEL_NAME,
        EMBEDDING_MODEL_PATH,
        DEVICE,
        EMBEDDING_BATCH_SIZE,
        HF_TOKEN
    ).embedding_instance()
    
    vector_store = VecStore(embedding_model, DB_PATH)
    vector_store.make_store()
    
    chat_model = ChatModel(
        CHAT_MODEL_NAME,
        CHAT_MODEL_PATH,
        DEVICE,
        CHAT_BATCH_SIZE,
        HF_TOKEN,
        MODEL_TASK,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        TOP_P,
        REPETITION_PENALTY,
        QUANTIZATION_CONFIG
    ).chat_instance()
    
    chain = Chain(preprocessor, vector_store, chat_model)

    application.add_handler(MessageHandler(filters.Text(START_BUTTON), start))
    application.add_handler(MessageHandler(filters.Text(HELP_BUTTON), help))

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Text(ADD_DATA_BUTTON), add_data_command)],
        states={
            ADDING_DATA : [
                MessageHandler(filters.Text(CANCEL_BUTTON), cancel),
                MessageHandler(filters.TEXT | filters.Document.ALL, partial(add_data, chain=chain)),
            ]
        },
        fallbacks=[MessageHandler(filters.ALL, fallback)]
    )
    application.add_handler(conv_handler)

    application.add_handler(
        MessageHandler(
            filters.TEXT, 
            partial(question_answer, chain=chain)
        )
    )

    application.run_polling()


if __name__ == "__main__":
    main()
