#include "sdetai/config.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <filesystem>

namespace sdetai {

namespace fs = std::filesystem;

void init_logger(const fs::path& log_dir, bool console_output, bool file_output) {
    std::vector<spdlog::sink_ptr> sinks;
    
    if (console_output) {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    }
    
    if (file_output && !log_dir.empty()) {
        fs::create_directories(log_dir);
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            (log_dir / "sdetai.log").string(), 1024 * 1024 * 10, 5
        );
        sinks.push_back(file_sink);
    }
    
    auto logger = std::make_shared<spdlog::logger>("sdetai", sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::debug);
    logger->flush_on(spdlog::level::debug);
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
}

Logger& get_logger() {
    static auto logger = spdlog::get("sdetai");
    if (!logger) {
        init_logger("", true, false);
        logger = spdlog::get("sdetai");
    }
    return *logger;
}

} // namespace sdetai