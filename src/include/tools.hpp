#pragma once

#include "sdetai/config.hpp"
#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <functional>

namespace sdetai {

class FileTools {
public:
    explicit FileTools(const AgentConfig& config);
    
    // Read file content
    ToolResult read_file(const fs::path& path);
    
    // Write file (creates directories if needed)
    ToolResult write_file(const fs::path& path, const std::string& content);
    
    // Apply multiple edits to a file
    ToolResult edit_file(const fs::path& path, const std::vector<FileEdit>& edits);
    
    // List files in directory
    ToolResult list_files(const fs::path& path, bool recursive = false);
    
    // Glob pattern matching
    ToolResult glob_files(const std::string& pattern);
    
    // Get file info
    ToolResult file_info(const fs::path& path);
    
    // Search in files (grep)
    ToolResult grep_files(const std::string& pattern, const fs::path& root = "");
    
    // Read file with line numbers
    ToolResult read_file_lines(const fs::path& path, std::optional<int> start_line = std::nullopt, std::optional<int> end_line = std::nullopt);

private:
    const AgentConfig& config_;
    
    bool is_allowed_path(const fs::path& path) const;
    fs::path resolve_path(const fs::path& path) const;
    bool matches_ignore(const fs::path& path) const;
};

} // namespace sdetai