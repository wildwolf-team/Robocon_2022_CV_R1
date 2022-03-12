#pragma once

#include <fmt/core.h>

#include <chrono>
#include <fstream>

namespace Log {
auto info_fm = fmt::format(fg(fmt::color::green), "INFO");
auto warn_fm = fmt::format(fg(fmt::color::yellow), "WARN");
auto error_fm = fmt::format(fg(fmt::color::red), "ERROR");
class log {
 public:
  log(const std::string &log_file_path) {
    log_file.open(log_file_path, std::ios::app);
  }

  void LogOut(const std::string log_topic, const std::string &message,
              const std::string &fm , bool print_flag) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::system_clock::now().time_since_epoch());
    log_file << "[" << ms.count() << "]"
             << "[" << log_topic << "]"
             << "[" << fm.c_str() << "]" << message << std::endl;
    if (print_flag) fmt::print("[{}][{}][{}]{}\n", ms.count(),
                               log_topic , fm, message);
  }

 private:
  std::ofstream log_file;
};
}  // namespace Log