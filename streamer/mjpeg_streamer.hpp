#pragma once
#include <sstream>

#include "streamer_impl.hpp"

class RoboStreamer : public nadjieb::MJPEGStreamer {
 public:
  void callOutFunc(nadjieb::net::HTTPRequest &req) {
    func_callback_(req);
  }

  void setCallbackFuncPtr(std::function<void(nadjieb::net::HTTPRequest &req)> func_ptr) {
    func_callback_ = func_ptr;
  }

  template <typename T>
  void publish_text_value(const std::string &div_id, const T &value,
                          const std::string &path = "/iframe") {
    std::string buffer =
      "<script type=\"text/javascript\">"
      "parent.document.getElementById(\'" +
      div_id + "\').innerHTML = \"" +
      std::to_string(value) + "\"</script>";
    this->publish(path, buffer);
  }

  template <typename T>
  void publish_charts_value(const std::string &chart_js_class_name,
                            const T &value,
                            const std::string &path = "/iframe") {
    std::string buffer =
      "<script>"
      "window.parent." +
      chart_js_class_name +
      ".update(" + std::to_string(value) +
      ");</script>";
    this->publish(path, buffer);
  }

template <typename... Args>
  void publish_console_log(const Args&... message) {
    const std::string path = "/iframe";
    std::ostringstream ss;
    using List= int[];
    (void)List{0, ( (void)(ss << message), 0 ) ... };
    std::string buffer =
      "<script type=\"text/javascript\">"
      "var now=new Date();"
      "window.parent.console.log(now.toISOString(),\'," +
      ss.str() +
      // message + "\');</script>";
      "\');</script>";
    this->publish(path, buffer);
  }

  void call_html_js_function(const std::string &func_name,
                             const std::string &path = "/iframe") {
    std::string buffer =
      "<script type=\"text/javascript\">"
      "window.parent." +
      func_name + "</script>";
    this->publish(path, buffer);
  }

 private:
  std::function<void(nadjieb::net::HTTPRequest &req)> func_callback_;
};