from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler
from src.settings import (
    START_STR, HELP_STR, CANCEL_STR, 
    ADD_DATA_STR, ADDING_DATA,
    HANDLERS_LOG_PATH
)
from src.logger import AppLogger
from io import BytesIO


logger = AppLogger(__name__, HANDLERS_LOG_PATH)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[START_STR, HELP_STR, ADD_DATA_STR]]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Добрый день! Данный бот представляет из себя вопросно-ответную систему.\n\n"
        "Вы можете задавать вопросы, а также добавлять информацию в базу в "
        "виде документов форматов txt, pdf, doc/docx, а также в виде текста.",
        reply_markup=markup
    )
    logger.debug("Start handler")


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[START_STR, HELP_STR, ADD_DATA_STR]]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Справка:\n\n"
        "Кнопка Старт - начало общения/запуск бота\n"
        "Кнопка Помощь - вызов подсказки\n"
        "Кнопка Добавить данные - возможность добавления информации текстом или документом "
        "в следующем сообщении\n\n"
        "При использовании текстовых сообщений в чате будет вызываться модель для общения "
        "и ответов на вопросы",
        reply_markup=markup
    )
    logger.debug("Help handler")


async def add_data_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[CANCEL_STR]]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Пожалуйста, отправьте текст",
        reply_markup=markup
    )
    logger.debug("Add data handler")
    return ADDING_DATA


async def add_data(update: Update, context: ContextTypes.DEFAULT_TYPE, chain):
    message = update.message

    if bool(message.document):
        document = message.document
        file_name = document.file_name
        file = await document.get_file()
        file_bytes = await file.download_as_bytearray()
        chain.load_data(BytesIO(file_bytes), file_name)
    elif bool(message.text):
        chain.load_data(message.text)

    keyboard = [[START_STR, HELP_STR, ADD_DATA_STR]]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Данные добавлены в базу",
        reply_markup=markup
    )
    logger.debug("Add data")

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[START_STR, HELP_STR, ADD_DATA_STR]]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Добавление данных в базу успешно отменено",
        reply_markup=markup
    )
    logger.debug("Cancel handler")
    return ConversationHandler.END


async def question_answer(update: Update, context: ContextTypes.DEFAULT_TYPE, chain):
    text = update.message.text
    reply = chain.invoke(text)
    keyboard = [[START_STR, HELP_STR, ADD_DATA_STR]]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    await update.message.reply_text(reply, reply_markup=markup)

    logger.debug("Answering handler")


async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[START_STR, HELP_STR, ADD_DATA_STR]]
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    await update.message.reply_text("Произошла ошибка", reply_markup=markup)

    logger.debug("Fallback handler")
    