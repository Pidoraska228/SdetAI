#pragma once

#include <string>
#include <filesystem>
#include <vector>
#include <optional>
#include <fstream>
#include <system_error>
#include <algorithm>
#include <regex>
#include <sstream>

namespace sdetai {

namespace fs = std::filesystem;

// Windows-specific path handling utilities
class FileSystem {
public:
    // Normalize path for current platform (handles Windows backslashes)
    static fs::path normalize(const fs::path& path) {
        return fs::weakly_canonical(path);
    }
    
    // Get absolute path, resolving relative paths against workspace root
    static fs::path resolve(const fs::path& workspace_root, const fs::path& path) {
        if (path.is_absolute()) {
            return normalize(path);
        }
        return normalize(workspace_root / path);
    }
    
    // Check if path is within workspace (security boundary)
    static bool is_within_workspace(const fs::path& workspace_root, const fs::path& path) {
        try {
            auto abs_workspace = normalize(workspace_root);
            auto abs_path = normalize(path);
            
            // On Windows, compare case-insensitively
            #ifdef _WIN32
            auto workspace_str = abs_workspace.generic_string();
            auto path_str = abs_path.generic_string();
            std::transform(workspace_str.begin(), workspace_str.end(), workspace_str.begin(), ::tolower);
            std::transform(path_str.begin(), path_str.end(), path_str.begin(), ::tolower);
            return path_str.rfind(workspace_str, 0) == 0;
            #else
            return abs_path.string().rfind(abs_workspace.string(), 0) == 0;
            #endif
        } catch (const fs::filesystem_error&) {
            return false;
        }
    }
    
    // Read entire file as string (UTF-8)
    static std::optional<std::string> read_file(const fs::path& path) {
        try {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) return std::nullopt;
            
            auto size = file.tellg();
            if (size <= 0) return std::string();
            
            file.seekg(0);
            std::string content(size, '\0');
            file.read(content.data(), size);
            return content;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
    
    // Write file atomically (write to temp, then rename)
    static bool write_file(const fs::path& path, const std::string& content, bool create_dirs = true) {
        try {
            if (create_dirs) {
                fs::create_directories(path.parent_path());
            }
            
            fs::path temp_path = path;
            temp_path += ".tmp";
            
            std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
            if (!file) return false;
            
            file.write(content.data(), content.size());
            if (!file) return false;
            file.close();
            
            fs::rename(temp_path, path);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }
    
    // List files in directory (non-recursive)
    static std::vector<fs::path> list_files(const fs::path& dir, bool recursive = false) {
        std::vector<fs::path> files;
        try {
            if (recursive) {
                for (auto& entry : fs::recursive_directory_iterator(dir)) {
                    if (entry.is_regular_file()) {
                        files.push_back(entry.path());
                    }
                }
            } else {
                for (auto& entry : fs::directory_iterator(dir)) {
                    if (entry.is_regular_file()) {
                        files.push_back(entry.path());
                    }
                }
            }
        } catch (const fs::filesystem_error&) {}
        return files;
    }
    
    // Glob pattern matching
    static std::vector<fs::path> glob(const fs::path& root, const std::string& pattern) {
        std::vector<fs::path> results;
        try {
            // Convert glob to regex: . -> \. , * -> .* , ? -> .
            std::string regex_pattern;
            for (char c : pattern) {
                if (c == '.') regex_pattern += "\\.";
                else if (c == '*') regex_pattern += ".*";
                else if (c == '?') regex_pattern += ".";
                else regex_pattern += c;
            }
            std::regex re(regex_pattern);
            
            for (auto& entry : fs::recursive_directory_iterator(root)) {
                if (entry.is_regular_file()) {
                    auto rel_path = fs::relative(entry.path(), root).generic_string();
                    if (std::regex_match(rel_path, re)) {
                        results.push_back(entry.path());
                    }
                }
            }
        } catch (const std::exception&) {}
        return results;
    }
    
    // Get file size
    static std::optional<uintmax_t> file_size(const fs::path& path) {
        try {
            return fs::file_size(path);
        } catch (const fs::filesystem_error&) {
            return std::nullopt;
        }
    }
    
    // Check if path exists and is a file
    static bool is_file(const fs::path& path) {
        return fs::exists(path) && fs::is_regular_file(path);
    }
    
    // Check if path exists and is a directory
    static bool is_directory(const fs::path& path) {
        return fs::exists(path) && fs::is_directory(path);
    }
    
    // Create directory (including parents)
    static bool create_directories(const fs::path& path) {
        try {
            return fs::create_directories(path);
        } catch (const fs::filesystem_error&) {
            return false;
        }
    }
    
    // Get last write time
    static std::optional<fs::file_time_type> last_write_time(const fs::path& path) {
        try {
            return fs::last_write_time(path);
        } catch (const fs::filesystem_error&) {
            return std::nullopt;
        }
    }
    
    // Read file lines with line numbers
    static std::vector<std::pair<int, std::string>> read_lines(const fs::path& path) {
        std::vector<std::pair<int, std::string>> lines;
        auto content = read_file(path);
        if (!content) return lines;
        
        int line_num = 1;
        std::string line;
        std::istringstream stream(*content);
        while (std::getline(stream, line)) {
            lines.emplace_back(line_num++, line);
        }
        return lines;
    }
    
    // Find project root (look for CMakeLists.txt, .git, package.json, etc.)
    static std::optional<fs::path> find_project_root(const fs::path& start) {
        static const std::vector<std::string> markers = {
            "CMakeLists.txt", ".git", "package.json", "pom.xml", "build.gradle",
            "Cargo.toml", "go.mod", "pyproject.toml", "setup.py", "Makefile"
        };
        
        fs::path current = normalize(start);
        while (true) {
            for (const auto& marker : markers) {
                if (fs::exists(current / marker)) {
                    return current;
                }
            }
            auto parent = current.parent_path();
            if (parent == current) break;
            current = parent;
        }
        return std::nullopt;
    }
};

} // namespace sdetai