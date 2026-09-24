import os

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

from Memory.Memorycontroller import MemoryController


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")


import socks
import socket

socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 10808)
socket.socket = socks.socksocket


controller = MemoryController(r"Config.json")

BOT_USERNAME = os.getenv("BOT_USERNAME", "NXFIGOVbot")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    text = message.text or ""
    chat_type = message.chat.type
    sender = message.from_user.first_name
    username = message.from_user.username or str(message.from_user.id)
    user_id = str(message.from_user.id)

    print(f"New message from {sender} in {chat_type}: {text}")

    if chat_type in ["group", "supergroup"]:

        if f"@{BOT_USERNAME}" in text:

            response = await controller.Agent_process(username, text, user_id)
            await message.reply_text(response)

        elif (
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.username == BOT_USERNAME
        ):

            response = await controller.Agent_process(username, text, user_id)
            await message.reply_text(response)

        elif text.lower().startswith("/"):
            pass

        else:
            print(f"Ignoring normal message in group: {text}")

    else:

        response = await controller.Agent_process(username, text, user_id)
        await message.reply_text(response)


async def say(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = " ".join(context.args)
    if not msg:
        await update.message.reply_text("INPUT ERROR")
        return

    chat_type = update.message.chat.type
    username = update.message.from_user.username or str(update.message.from_user.id)
    user_id = str(update.message.from_user.id)

    response = await controller.Agent_process(username, msg, user_id)

    if chat_type in ["group", "supergroup"]:
        await update.message.reply_text(
            f"@{update.message.from_user.username} said: {response}"
        )
    else:
        await update.message.reply_text(response)


def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("say", say))

    print("Bot is running...")
    print(f"Bot username: @{BOT_USERNAME}")
    print("Bot is ready for groups and private chats!")

    app.run_polling()


if __name__ == "__main__":
    main()
