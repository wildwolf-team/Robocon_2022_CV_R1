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
  log(const std::string &log_file_path, const std::string log_flag_) {
    log_file.open(log_file_path, std::ios::app);
    log_flag = log_flag_;
  }
  void LogOut(const std::string &message, const std::string &fm , bool print_flag) {
    log_file << "[" << log_flag << "]"
             << "[" << fm << "]" << message << std::endl;
    if (print_flag) fmt::print("[{}]{}\n", fm, message);
  }

 private:
  std::ofstream log_file;
  std::string log_flag;
};
}  // namespace Log