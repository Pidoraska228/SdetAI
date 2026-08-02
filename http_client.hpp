#pragma once
// =============================================================================
// HTTP Client for llama-server communication
// Windows: WinHTTP, Linux: libcurl (if available)
// =============================================================================

#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <chrono>

namespace sdetai {

struct HttpResponse {
    int status_code = 0;
    std::string body;
    std::string error;
    bool success() const { return status_code >= 200 && status_code < 300; }
};

struct HttpClient {
    HttpClient() = default;
    ~HttpClient() = default;
    
    // POST JSON to endpoint
    HttpResponse post_json(
        const std::string& url,
        const std::string& json_body,
        const std::vector<std::pair<std::string, std::string>>& headers = {},
        int timeout_seconds = 30
    );
    
    // POST JSON with streaming callback (for token-by-token)
    bool post_json_stream(
        const std::string& url,
        const std::string& json_body,
        std::function<bool(const std::string& token)> callback,
        const std::vector<std::pair<std::string, std::string>>& headers = {},
        int timeout_seconds = 60
    );
    
    // GET request
    HttpResponse get(
        const std::string& url,
        const std::vector<std::pair<std::string, std::string>>& headers = {},
        int timeout_seconds = 10
    );
    
    // Check if llama-server is running
    bool check_server(const std::string& base_url = "http://localhost:8080");
};

} // namespace sdetai