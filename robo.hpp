#pragma once
#include <thread>
#include <chrono>
#include <exception>
#include <cmath>

#include <opencv2/opencv.hpp>
#define FMT_HEADER_ONLY
#include <fmt/color.h>
#include <fmt/core.h>

#include "angle/prediction/kalman_prediction.hpp"
#include "angle/solvePnP/solvePnP.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "devices/new_serial/serial.hpp"
#ifdef USE_OV_DETECTOR
#include "OpenVINO/ovdetector.hpp"
#endif
#include "robo_data.h"
#include "streamer/mjpeg_streamer.hpp"
#include "ThreadPool.h"
#ifdef USE_TRT_DETECTOR
#include "TensorRTx/trtdetector.hpp"
#endif
#include "utils/json.hpp"
#include "utils/simple_cpp_sockets.hpp"

class RoboR1 {
 private:
  bool end_node_{false};
  bool is_kalman_open_{true};
  bool debug_mode{true};

  nlohmann::json debug_info_;

  RoboCmd robo_cmd;
  RoboInf robo_inf;
  nlohmann::json config_json;

  std::mutex mtx;

  std::unique_ptr<RoboSerial> serial_;
  std::shared_ptr<mindvision::VideoCapture> camera_;

#ifdef USE_TRT_DETECTOR
  std::unique_ptr<TRTDetector> yolo_detection_;
#endif

#ifdef USE_OV_DETECTOR
  std::unique_ptr<Detector> yolo_ov_detector_;
#endif

  std::unique_ptr<solvepnp::PnP> pnp_;
  std::unique_ptr<KalmanPrediction> kalman_prediction_;
  std::unique_ptr<RoboStreamer> streamer_;
  std::unique_ptr<UDPClient> pj_udp_cl_;

 public:
  RoboR1();
  void Init();
  void Spin();

  void uartWrite();
  void uartRead();

  void streamerCallback(const nadjieb::net::HTTPRequest &req);

  void detectionTask();
  void targetFilter(std::vector<myrobo::detection> &_pred,
    cv::Rect &_target, bool &_is_lose);
  void storeRoboInfo(const cv::Point2f &_pnp_angle, const float &_depth,
    const bool &_detect_object);

  void stop();
  ~RoboR1();
};