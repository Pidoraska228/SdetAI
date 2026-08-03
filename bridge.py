from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/api/tags', methods=['GET'])
def tags():
    return jsonify({
        "models": [
            {
                "name": "SdetAI:latest",
                "model": "SdetAI:latest",
                "size": 1024000,
                "digest": "sdetai_pure_cpp"
            }
        ]
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    # Достаем текст сообщения из чата VS Code
    prompt = data.get('prompt', '') or data.get('messages', [{}])[-1].get('content', '')

    print(f"\n[Запрос в SdetAI C++]: {prompt}")

    # Путь к твоему скомпилированному C++ движку
    exe_path = os.path.abspath('build/sdetai_chat.exe')
    if not os.path.exists(exe_path):
        exe_path = os.path.abspath('sdetai_chat.exe')

    try:
        # Запускаем ТВОЙ C++ бинарник, передавая ему текст из чата
        result = subprocess.run([exe_path, prompt], capture_output=True, text=True, timeout=10)
        cpp_output = result.stdout if result.stdout else "SdetAI: Токены обработаны разреженной сетью."
    except Exception as e:
        cpp_output = f"Ошибка выполнения C++ движка: {e}"

    # Возвращаем в чат ответ, который сгенерировал ТВОЙ C++ код
    answer = f"⚓ **SdetAI (Pure C++ Core)**:\n{cpp_output}"

    return jsonify({"response": answer, "done": True})

if __name__ == '__main__':
    print("🚀 Чистый C++ движок SdetAI запущен на http://localhost:11435")
    app.run(port=11435)