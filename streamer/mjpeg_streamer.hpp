#pragma once
#include <sstream>

#include "streamer_impl.hpp"

class RoboStreamer : public nadjieb::MJPEGStreamer {
 public:
  void callOutFunc(nadjieb::net::HTTPRequest &req) {
    if (req.getTarget() == "/stop") {
      stop_node_func_();
    }
    if ((req.getTarget() == "/setCameraExposure") && (req.getMethod() == "POST")){
      camera_set_exposure_func_(stoi(req.getBody()));
    }
  }

  void setCameraSetExposureFuncPtr(std::function<void(int)> func_ptr) {
    camera_set_exposure_func_ = func_ptr;
  }

  void setStopNodeFuncPtr(std::function<void(void)> func_ptr) {
    stop_node_func_ = func_ptr;
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
      func_name + "()</script>";
    this->publish(path, buffer);
  }

 private:
  std::function<void(int)> camera_set_exposure_func_;
  std::function<void(void)> stop_node_func_;
};