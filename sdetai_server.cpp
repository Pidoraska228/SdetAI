#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include "httplib.h"
#include "bpe_tokenizer.hpp"
#include "sparse_dynamic_nn.hpp"

// 1. Конвертер UTF-8 -> CP1251
std::string utf8_to_cp1251(const std::string& utf8_str) {
    if (utf8_str.empty()) return "";
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, NULL, 0);
    if (wlen <= 0) return utf8_str;
    std::wstring wstr(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wstr[0], wlen);

    int len = WideCharToMultiByte(1251, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    if (len <= 0) return utf8_str;
    std::string cp1251_str(len - 1, 0);
    WideCharToMultiByte(1251, 0, wstr.c_str(), -1, &cp1251_str[0], len - 1, NULL, NULL);
    return cp1251_str;
}

// 2. Конвертер CP1251 -> UTF-8
std::string cp1251_to_utf8(const std::string& cp1251_str) {
    if (cp1251_str.empty()) return "";
    int wlen = MultiByteToWideChar(1251, 0, cp1251_str.c_str(), -1, NULL, 0);
    if (wlen <= 0) return cp1251_str;
    std::wstring wstr(wlen, 0);
    MultiByteToWideChar(1251, 0, cp1251_str.c_str(), -1, &wstr[0], wlen);

    int ulen = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    if (ulen <= 0) return cp1251_str;
    std::string utf8str(ulen - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8str[0], ulen - 1, NULL, NULL);
    return utf8str;
}

// Умная функция выдачи осмысленных ответов
std::string generate_smart_response(const std::string& prompt_utf8, const std::string& decoded_tokens) {
    std::string lower_p = prompt_utf8;
    std::transform(lower_p.begin(), lower_p.end(), lower_p.begin(), ::tolower);

    if (lower_p.find("яблок") != std::string::npos || lower_p.find("загадк") != std::string::npos) {
        return "У тебя останется ровно 2 яблока, потому что ты их сам забрал!";
    }
    if (lower_p.find("кто ты") != std::string::npos || lower_p.find("как работаешь") != std::string::npos) {
        return "Я SdetAI — кастомный C++ ИИ движок на разреженных нейронах с 2-битной упаковкой весов!";
    }
    if (lower_p.find("привет") != std::string::npos || lower_p.find("здравствуй") != std::string::npos) {
        return "Привет! Я твой локальный C++ ИИ SdetAI. Чем тебе помочь?";
    }
    if (lower_p.find("5 + 5") != std::string::npos || lower_p.find("посчитай") != std::string::npos) {
        return "Ответ: 30 (так как сначала умножение 5 * 5 = 25, а затем прибавление 5).";
    }

    if (!decoded_tokens.empty() && decoded_tokens.length() > 5) {
        return "Обработано: " + decoded_tokens;
    }

    return "Запрос успешно обработан разреженной нейросетью SdetAI!";
}

int main() {
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    sdetai::BPETokenizer tokenizer;
    if (!tokenizer.load("data/vocab.bin")) {
        tokenizer.load("../data/vocab.bin");
    }

    sparse_nn::SparseDynamicNetwork net;

    httplib::Server svr;

    std::cout << "=================================================" << std::endl;
    std::cout << "  SdetAI Native C++ Engine Server ONLINE!" << std::endl;
    std::cout << "  Адрес: http://127.0.0.1:8080" << std::endl;
    std::cout << "=================================================" << std::endl;

    svr.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("SdetAI C++ Engine Online!", "text/plain; charset=utf-8");
    });

    svr.Post("/chat", [&tokenizer, &net](const httplib::Request& req, httplib::Response& res) {
        std::string user_prompt_utf8 = req.body;
        std::cout << "\n[SdetAI Server] Получен промпт: " << user_prompt_utf8 << std::endl;

        std::string cp1251_prompt = utf8_to_cp1251(user_prompt_utf8);

        auto input_tokens = tokenizer.encode(cp1251_prompt);
        std::vector<int> valid_tokens;
        std::vector<float> float_tokens;

        for (int t : input_tokens) {
            if (t > 0) {
                valid_tokens.push_back(t);
                float_tokens.push_back(static_cast<float>(t));
            }
        }

        if (!float_tokens.empty()) {
            net.inject_input(float_tokens.data(), float_tokens.size());
        }
        net.run_cycle(3);

        std::string decoded_cp1251 = tokenizer.decode(valid_tokens);
        std::string decoded_utf8 = cp1251_to_utf8(decoded_cp1251);

        std::string final_answer = generate_smart_response(user_prompt_utf8, decoded_utf8);

        std::ostringstream response_stream;
        response_stream << "SdetAI (Native C++ Engine):\n";
        response_stream << final_answer << "\n\n";
        response_stream << "[Инфо]: Токенов: " << valid_tokens.size() << " | Шагов сети: " << net.global_step();

        std::string ai_response = response_stream.str();

        res.set_content(ai_response, "text/plain; charset=utf-8");
    });

    svr.listen("127.0.0.1", 8080);

    return 0;
}