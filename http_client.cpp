#include "http_client.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#else
#include <curl/curl.h>
#endif

#include <vector>
#include <sstream>
#include <chrono>
#include <thread>

namespace sdetai {

#ifdef _WIN32

static std::wstring to_wide(const std::string& s) {
    if (s.empty()) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], len);
    return w;
}

static std::string to_utf8(const std::wstring& w) {
    if (w.empty()) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(len, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], len, nullptr, nullptr);
    return s;
}

HttpResponse HttpClient::post_json(
    const std::string& url,
    const std::string& json_body,
    const std::vector<std::pair<std::string, std::string>>& headers,
    int timeout_seconds
) {
    HttpResponse resp;
    
    // Parse URL
    std::wstring wurl = to_wide(url);
    URL_COMPONENTSW comps = {};
    comps.dwStructSize = sizeof(comps);
    comps.dwSchemeLength = 1;
    comps.dwHostNameLength = 1;
    comps.dwUrlPathLength = 1;
    comps.dwExtraInfoLength = 1;
    
    WinHttpCrackUrl(wurl.c_str(), 0, ICU_ESCAPE, &comps);
    
    std::wstring scheme(comps.lpszScheme, comps.dwSchemeLength);
    std::wstring host(comps.lpszHostName, comps.dwHostNameLength);
    std::wstring path(comps.lpszUrlPath, comps.dwUrlPathLength);
    if (comps.dwExtraInfoLength > 0) {
        path += std::wstring(comps.lpszExtraInfo, comps.dwExtraInfoLength);
    }
    INTERNET_PORT port = comps.nPort;
    bool secure = (scheme == L"https");
    
    HINTERNET hSession = WinHttpOpen(L"SdetAI/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) { resp.error = "WinHttpOpen failed"; return resp; }
    
    HINTERNET hConnect = WinHttpConnect(hSession, host.c_str(), port, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); resp.error = "WinHttpConnect failed"; return resp; }
    
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", path.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0);
    if (!hRequest) { WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); resp.error = "WinHttpOpenRequest failed"; return resp; }
    
    // Set timeouts
    int timeout_ms = timeout_seconds * 1000;
    WinHttpSetTimeouts(hRequest, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
    
    // Build headers
    std::wstring header_str = L"Content-Type: application/json\r\n";
    for (const auto& [k, v] : headers) {
        header_str += to_wide(k) + L": " + to_wide(v) + L"\r\n";
    }
    
    std::wstring wbody = to_wide(json_body);
    
    BOOL sent = WinHttpSendRequest(hRequest, header_str.c_str(), -1, (LPVOID)wbody.c_str(), (DWORD)wbody.size() * 2, (DWORD)wbody.size() * 2, 0);
    if (!sent) { 
        WinHttpCloseHandle(hRequest); 
        WinHttpCloseHandle(hConnect); 
        WinHttpCloseHandle(hSession); 
        resp.error = "WinHttpSendRequest failed"; 
        return resp; 
    }
    
    if (!WinHttpReceiveResponse(hRequest, nullptr)) {
        WinHttpCloseHandle(hRequest); 
        WinHttpCloseHandle(hConnect); 
        WinHttpCloseHandle(hSession); 
        resp.error = "WinHttpReceiveResponse failed"; 
        return resp; 
    }
    
    // Get status code
    DWORD status_size = sizeof(DWORD);
    WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &resp.status_code, &status_size, WINHTTP_NO_HEADER_INDEX);
    
    // Read body
    std::string body;
    DWORD avail = 0;
    do {
        avail = 0;
        WinHttpQueryDataAvailable(hRequest, &avail);
        if (avail > 0) {
            std::vector<char> buf(avail);
            DWORD read = 0;
            if (WinHttpReadData(hRequest, buf.data(), avail, &read)) {
                body.append(buf.data(), read);
            }
        }
    } while (avail > 0);
    
    resp.body = body;
    
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    
    return resp;
}

bool HttpClient::post_json_stream(
    const std::string& url,
    const std::string& json_body,
    std::function<bool(const std::string& token)> callback,
    const std::vector<std::pair<std::string, std::string>>& headers,
    int timeout_seconds
) {
    // For streaming, we'd need chunked transfer encoding support
    // Simplified: just do regular POST and parse SSE manually
    auto resp = post_json(url, json_body, headers, timeout_seconds);
    if (!resp.success()) return false;
    
    // Parse SSE (Server-Sent Events) format
    std::istringstream iss(resp.body);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.rfind("data: ", 0) == 0) {
            std::string data = line.substr(6);
            if (data == "[DONE]") break;
            // Parse JSON to extract token
            // For now, assume plain text tokens
            if (!callback(data)) return false;
        }
    }
    return true;
}

HttpResponse HttpClient::get(
    const std::string& url,
    const std::vector<std::pair<std::string, std::string>>& headers,
    int timeout_seconds
) {
    HttpResponse resp;
    
    std::wstring wurl = to_wide(url);
    URL_COMPONENTSW comps = {};
    comps.dwStructSize = sizeof(comps);
    comps.dwSchemeLength = 1;
    comps.dwHostNameLength = 1;
    comps.dwUrlPathLength = 1;
    comps.dwExtraInfoLength = 1;
    
    WinHttpCrackUrl(wurl.c_str(), 0, ICU_ESCAPE, &comps);
    
    std::wstring scheme(comps.lpszScheme, comps.dwSchemeLength);
    std::wstring host(comps.lpszHostName, comps.dwHostNameLength);
    std::wstring path(comps.lpszUrlPath, comps.dwUrlPathLength);
    if (comps.dwExtraInfoLength > 0) {
        path += std::wstring(comps.lpszExtraInfo, comps.dwExtraInfoLength);
    }
    INTERNET_PORT port = comps.nPort;
    bool secure = (scheme == L"https");
    
    HINTERNET hSession = WinHttpOpen(L"SdetAI/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) { resp.error = "WinHttpOpen failed"; return resp; }
    
    HINTERNET hConnect = WinHttpConnect(hSession, host.c_str(), port, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); resp.error = "WinHttpConnect failed"; return resp; }
    
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0);
    if (!hRequest) { WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); resp.error = "WinHttpOpenRequest failed"; return resp; }
    
    int timeout_ms = timeout_seconds * 1000;
    WinHttpSetTimeouts(hRequest, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
    
    std::wstring header_str;
    for (const auto& [k, v] : headers) {
        header_str += to_wide(k) + L": " + to_wide(v) + L"\r\n";
    }
    
    BOOL sent = WinHttpSendRequest(hRequest, header_str.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : header_str.c_str(), -1, WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (!sent) { 
        WinHttpCloseHandle(hRequest); 
        WinHttpCloseHandle(hConnect); 
        WinHttpCloseHandle(hSession); 
        resp.error = "WinHttpSendRequest failed"; 
        return resp; 
    }
    
    if (!WinHttpReceiveResponse(hRequest, nullptr)) {
        WinHttpCloseHandle(hRequest); 
        WinHttpCloseHandle(hConnect); 
        WinHttpCloseHandle(hSession); 
        resp.error = "WinHttpReceiveResponse failed"; 
        return resp; 
    }
    
    DWORD status_size = sizeof(DWORD);
    WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &resp.status_code, &status_size, WINHTTP_NO_HEADER_INDEX);
    
    std::string body;
    DWORD avail = 0;
    do {
        avail = 0;
        WinHttpQueryDataAvailable(hRequest, &avail);
        if (avail > 0) {
            std::vector<char> buf(avail);
            DWORD read = 0;
            if (WinHttpReadData(hRequest, buf.data(), avail, &read)) {
                body.append(buf.data(), read);
            }
        }
    } while (avail > 0);
    
    resp.body = body;
    
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    
    return resp;
}

bool HttpClient::check_server(const std::string& base_url) {
    auto resp = get(base_url + "/health", {}, 5);
    return resp.success() && resp.status_code == 200;
}

#else
// Linux/macOS: libcurl implementation
struct CurlWriteData {
    std::string* buffer;
    std::function<bool(const std::string&)>* callback;
};

static size_t curl_write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    CurlWriteData* data = (CurlWriteData*)userp;
    size_t realsize = size * nmemb;
    if (data->callback && *data->callback) {
        std::string chunk((char*)contents, realsize);
        // Parse SSE
        std::istringstream iss(chunk);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.rfind("data: ", 0) == 0) {
                std::string d = line.substr(6);
                if (d == "[DONE]") return realsize;
                if (!(*data->callback)(d)) return 0;  // Stop
            }
        }
    } else if (data->buffer) {
        data->buffer->append((char*)contents, realsize);
    }
    return realsize;
}

HttpResponse HttpClient::post_json(...) {
    // Similar implementation using curl_easy_*
    return {};
}

bool HttpClient::post_json_stream(...) {
    // Use curl with write callback
    return {};
}

HttpResponse HttpClient::get(...) { return {}; }
bool HttpClient::check_server(const std::string& base_url) { return false; }

#endif

} // namespace sdetai