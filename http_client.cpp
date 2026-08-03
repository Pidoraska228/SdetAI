#include "http_client.hpp"
#include <iostream>

namespace sdetai {

HttpResponse HttpClient::post_json(const std::string& url, const std::string& data,
                                  const std::vector<std::pair<std::string, std::string>>& headers, int timeout) {
    return {};
}

bool HttpClient::post_json_stream(const std::string& url, const std::string& data,
                                 std::function<bool(const std::string&)> callback,
                                 const std::vector<std::pair<std::string, std::string>>& headers, int timeout) {
    return false;
}

HttpResponse HttpClient::get(const std::string& url,
                            const std::vector<std::pair<std::string, std::string>>& headers, int timeout) {
    return {};
}

bool HttpClient::check_server(const std::string& base_url) {
    return false;
}

} // namespace sdetai