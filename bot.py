import telebot
import requests

# 1. Вставь сюда токен своего бота от @BotFather:
BOT_TOKEN = "8476860472:AAGQSxIpj41DI2XW8mDjwwL--PCmiXeUh4Q"

bot = telebot.TeleBot(BOT_TOKEN)

print("=========================================")
print(" Telegram-бот SdetAI запущен и готов!")
print("=========================================")

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(
        message, 
        "Привет! Я SdetAI Bot. Я работаю на локальном нативном C++ движке.\nНапиши мне любой вопрос!"
    )

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text
    print(f"[Telegram] Получено сообщение: {user_text}")

    try:
        # Отправляем текст в твой C++ сервер на http://localhost:8080/chat
        res = requests.post(
            "http://localhost:8080/chat", 
            data=user_text.encode('utf-8'),
            headers={'Content-Type': 'text/plain; charset=utf-8'}
        )
        
        # Получаем ответ от C++ сервера
        ai_reply = res.text
        
        # Отправляем красивый ответ пользователю в Telegram
        bot.reply_to(message, ai_reply)

    except Exception as e:
        bot.reply_to(message, f"Ошибка подключения к C++ серверу: {str(e)}")

# Запуск постоянной работы бота
bot.polling(non_stop=True)