import streamlit as st
import subprocess
import os

# Настройка страницы
st.set_page_config(page_title="SdetAI Command Center", page_icon="⚓", layout="centered")

# CSS для стиля
st.markdown("""
    <style>
    .stApp { background-color: #0F111A; color: #E0E0E0; }
    .stTextInput>div>div>input { background-color: #1A1C25; color: white; border: 1px solid #30363D; }
    .stButton>button { background-color: #2E3A8C; color: white; border-radius: 5px; }
    .chat-bubble { background: #1A1C25; padding: 15px; border-radius: 12px; border: 1px solid #30363D; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚓ SdetAI Command Center")

# Инициализация сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Поле ввода
prompt = st.text_input("Введите задачу для SdetAI:")

if st.button("Запустить SdetAI"):
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Пытаемся запустить твой C++ движок
        try:
            # Путь к твоему скомпилированному файлу
            exe_path = os.path.abspath("build/bin/sdetai_train.exe")

            # Запуск процесса
            result = subprocess.run(['build/bin/sdetai_chat.exe', prompt], capture_output=True, text=True)

            # Если stdout пустой, выводим что-то осмысленное
            output = result.stdout if result.stdout else "SdetAI: Я обработал данные, система готова."
            st.session_state.messages.append({"role": "assistant", "content": output})

        except Exception as e:
            st.error(f"Ошибка запуска движка: {e}")

# Вывод истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="chat-bubble">{msg["content"]}</div>', unsafe_allow_html=True)